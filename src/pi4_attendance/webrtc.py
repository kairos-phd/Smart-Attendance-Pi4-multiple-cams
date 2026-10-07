from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
import time
import uuid
from fractions import Fraction
from typing import TYPE_CHECKING

import firebase_admin
from av import VideoFrame
from firebase_admin import credentials, db
from aiortc import (
    MediaStreamTrack,
    VideoStreamTrack,
    RTCPeerConnection,
    RTCConfiguration,
    RTCIceServer,
    RTCSessionDescription,
)

if TYPE_CHECKING:
    from .service import AttendanceRuntime

LOG = logging.getLogger(__name__)

STUN_SERVERS = [
    RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
]

POLL_INTERVAL = 0.5
ICE_GATHER_TIMEOUT = 10.0
SESSION_MAX_AGE = 300.0
STREAM_FPS = 3.0
SIGNALING_TIMEOUT = 10.0


class PiVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, runtime: AttendanceRuntime, camera_id: str) -> None:
        super().__init__()
        self.runtime = runtime
        self.camera_id = camera_id
        self._last_sequence = -1
        self._frame_index = 0
        self._next_frame_at = time.monotonic()

    async def recv(self) -> VideoFrame:
        if self.readyState != "live":
            raise RuntimeError("WebRTC video track ended")

        # Keep the WebRTC sender at a controlled rate instead of
        # duplicating camera frames as fast as the browser requests them.
        now = time.monotonic()
        wait = self._next_frame_at - now
        if wait > 0:
            await asyncio.sleep(wait)

        self._next_frame_at = max(
            self._next_frame_at + (1.0 / STREAM_FPS),
            time.monotonic(),
        )

        sample = getattr(self.runtime.preview, "latest_display_sample", self.runtime.preview.latest_sample)
        frame, sequence = sample(self.camera_id)

        # Camera frames are updated independently by CameraWorker.
        # Reuse the newest frame when there is no new sample yet.
        while frame is None and self.readyState == "live":
            await asyncio.sleep(0.02)
            frame, sequence = sample(self.camera_id)

        if frame is None:
            raise RuntimeError(
                f"No frame available for camera {self.camera_id}"
            )

        if sequence != self._last_sequence:
            self._last_sequence = sequence

        video_frame = VideoFrame.from_ndarray(
            frame,
            format="bgr24",
        )

        # aiortc / RTP requires monotonically increasing timestamps.
        video_frame.pts = self._frame_index
        video_frame.time_base = Fraction(1, int(STREAM_FPS))
        self._frame_index += 1

        if self._frame_index == 1 or self._frame_index % 30 == 0:
            LOG.info(
                "WebRTC video frames: camera=%s frames=%d sequence=%d",
                self.camera_id,
                self._frame_index,
                sequence,
            )

        return video_frame


class WebRTCManager:
    def __init__(self, runtime: AttendanceRuntime) -> None:
        self.runtime = runtime
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.peer_connections: dict[str, RTCPeerConnection] = {}
        self._peer_deadlines: dict[str, float] = {}
        self.max_peers = max(1, int(os.getenv("WEBRTC_MAX_PEERS", "3")))

        self.app = self._get_firebase_app()
        self.root_ref = db.reference("webrtc", app=self.app)
        self.status_ref = db.reference("status", app=self.app)

    @staticmethod
    def _get_firebase_app():
        try:
            return firebase_admin.get_app("sic-webrtc")
        except ValueError:
            service_account = os.getenv("FIREBASE_SERVICE_ACCOUNT", "").strip()
            project_id = os.getenv("FIREBASE_PROJECT_ID", "").strip()
            database_url = os.getenv("FIREBASE_DATABASE_URL", "").strip()

            if not service_account or not database_url:
                raise RuntimeError(
                    "FIREBASE_SERVICE_ACCOUNT and FIREBASE_DATABASE_URL are required"
                )

            options = {"databaseURL": database_url, "httpTimeout": SIGNALING_TIMEOUT}
            if project_id:
                options["projectId"] = project_id

            return firebase_admin.initialize_app(
                credentials.Certificate(service_account),
                options,
                name="sic-webrtc",
            )

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return

        self.thread = threading.Thread(
            target=self._thread_main,
            name="webrtc-signaling",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=5)

        self.thread = None

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception:
            LOG.exception("WebRTC signaling thread stopped unexpectedly")

    async def _run(self) -> None:
        self.loop = asyncio.get_running_loop()
        last_status_publish = 0.0
        watchdog = asyncio.create_task(self._expiry_watchdog())
        try:
            while not self.stop_event.is_set():
                try:
                    now = time.monotonic()
                    if now - last_status_publish >= 2.0:
                        await self._publish_status()
                        last_status_publish = now
                    await self._poll_sessions()
                except Exception:
                    LOG.exception("WebRTC signaling poll failed")
                await asyncio.sleep(POLL_INTERVAL)
        finally:
            watchdog.cancel()
            try:
                await watchdog
            except asyncio.CancelledError:
                pass
            await self._close_all()

    async def _remote(self, operation, *args):
        return await asyncio.wait_for(asyncio.to_thread(operation, *args), SIGNALING_TIMEOUT)

    async def _expire_local_peers(self) -> None:
        now = time.monotonic()
        for key, deadline in list(getattr(self, "_peer_deadlines", {}).items()):
            if now >= deadline:
                uid, session_id = key.split("/", 1)
                await self._cleanup_connection(uid, session_id)

    async def _expiry_watchdog(self) -> None:
        while not self.stop_event.is_set():
            await self._expire_local_peers()
            await asyncio.sleep(POLL_INTERVAL)

    def _session_for_this_pi(self, session: dict) -> bool:
        return session.get("pi_id") in (None, self.runtime.settings.pi_id)

    @staticmethod
    def _session_expired(session: dict) -> bool:
        created = session.get("created_at")
        if created is None:
            return False
        return (not isinstance(created, (int, float)) or not math.isfinite(created)
                or time.time() - created / 1000.0 >= SESSION_MAX_AGE)

    async def _publish_status(self) -> None:
        health = await asyncio.to_thread(self.runtime.health)

        cameras = []
        for row in health.get("camera_status", []):
            cameras.append(
                {
                    "camera_id": row.get("camera_id"),
                    "name": row.get("name"),
                    "status": row.get("status"),
                    "last_error": row.get("last_error"),
                    "last_seen": row.get("last_seen"),
                }
            )

        payload = {
            "pi_id": health.get("pi_id", self.runtime.settings.pi_id),
            "camera_count": len(self.runtime.preview.camera_ids),
            "cameras": cameras,
            "updated_at": int(time.time() * 1000),
        }

        await self._remote(
            self.status_ref.child(self.runtime.settings.pi_id).set,
            payload,
        )

    async def _poll_sessions(self) -> None:
        await self._expire_local_peers()
        snapshot = await self._remote(self.root_ref.get) or {}
        if not isinstance(snapshot, dict):
            raise ValueError("WebRTC signaling root must be a mapping")
        now = time.time()
        monotonic_now = time.monotonic()
        deadlines = getattr(self, "_peer_deadlines", {})
        for key in list(self.peer_connections):
            uid, session_id = key.split("/", 1)
            sessions = snapshot.get(uid, {})
            session = sessions.get(session_id) if isinstance(sessions, dict) else None
            created = session.get("created_at") if isinstance(session, dict) else None
            expired = isinstance(session, dict) and self._session_expired(session)
            if (not isinstance(session, dict) or session.get("state") in {"closed", "revoked", "error"}
                    or expired or monotonic_now >= deadlines.get(key, float("inf"))):
                await self._cleanup_connection(uid, session_id)

        for uid, user_sessions in snapshot.items():
            if not isinstance(user_sessions, dict):
                continue

            for session_id, session in list(user_sessions.items()):
                if not isinstance(session, dict):
                    continue
                if not self._session_for_this_pi(session):
                    continue

                if self._session_expired(session):
                    await self._delete_session(uid, session_id)
                    continue

                if session.get("answer") is not None:
                    continue
                if session.get("state") not in (None, "offer"):
                    continue

                offer = session.get("offer")
                camera_id = session.get("camera_id")

                if not isinstance(offer, dict):
                    continue
                if not isinstance(camera_id, str):
                    continue
                if camera_id not in self.runtime.preview.camera_ids:
                    await self._set_session_error(
                        uid,
                        session_id,
                        f"Unknown camera: {camera_id}",
                    )
                    continue

                if len(self.peer_connections) >= getattr(self, "max_peers", 3):
                    await self._set_session_error(uid, session_id, "WebRTC viewer capacity reached")
                    continue

                await self._handle_offer(
                    str(uid),
                    str(session_id),
                    camera_id,
                    offer,
                )

    async def _handle_offer(
        self,
        uid: str,
        session_id: str,
        camera_id: str,
        offer: dict,
    ) -> None:
        key = f"{uid}/{session_id}"

        if key in self.peer_connections:
            return

        session_ref = self.root_ref.child(uid).child(session_id)
        claim_id = uuid.uuid4().hex

        try:
            def claim(session):
                if (not isinstance(session, dict) or session.get("state") not in (None, "offer")
                        or session.get("answer") is not None or session.get("offer") != offer
                        or session.get("camera_id") != camera_id or not self._session_for_this_pi(session)
                        or self._session_expired(session)):
                    return session
                return {**session, "state": "answering", "pi_claim": claim_id}
            claimed = await self._remote(session_ref.transaction, claim)
            if not isinstance(claimed, dict) or claimed.get("pi_claim") != claim_id:
                return

            pc = RTCPeerConnection(
                RTCConfiguration(iceServers=STUN_SERVERS)
            )
            self.peer_connections[key] = pc
            created = claimed.get("created_at")
            remaining = SESSION_MAX_AGE if created is None else max(0.0, min(SESSION_MAX_AGE, created / 1000.0 + SESSION_MAX_AGE - time.time()))
            self._peer_deadlines[key] = time.monotonic() + remaining

            @pc.on("connectionstatechange")
            async def on_connectionstatechange() -> None:
                state = pc.connectionState
                LOG.info(
                    "WebRTC %s/%s connection state: %s",
                    uid,
                    session_id,
                    state,
                )

                if state in {"failed", "closed"}:
                    await self._cleanup_connection(uid, session_id)

            await pc.setRemoteDescription(
                RTCSessionDescription(
                    sdp=str(offer["sdp"]),
                    type=str(offer["type"]),
                )
            )

            pc.addTrack(PiVideoTrack(self.runtime, camera_id))

            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            await self._wait_for_ice_gathering(pc)

            local = pc.localDescription
            if local is None:
                raise RuntimeError("No local WebRTC description was generated")

            def publish(session):
                if (not isinstance(session, dict) or session.get("state") != "answering"
                        or session.get("pi_claim") != claim_id or session.get("offer") != offer
                        or session.get("camera_id") != camera_id or not self._session_for_this_pi(session)
                        or self._session_expired(session) or key not in self.peer_connections
                        or time.monotonic() >= self._peer_deadlines.get(key, 0)):
                    return session
                return {**session, "answer": {"type": local.type, "sdp": local.sdp}, "state": "answered"}
            published = await self._remote(session_ref.transaction, publish)
            if (not isinstance(published, dict) or published.get("pi_claim") != claim_id
                    or published.get("state") != "answered"):
                await self._cleanup_connection(uid, session_id)
                return

            LOG.info(
                "WebRTC answer published for camera %s, session %s",
                camera_id,
                session_id,
            )

        except Exception as exc:
            LOG.exception(
                "WebRTC offer handling failed for camera %s session %s",
                camera_id,
                session_id,
            )
            await self._cleanup_connection(uid, session_id)
            await self._set_session_error(uid, session_id, str(exc))

    async def _wait_for_ice_gathering(
        self,
        pc: RTCPeerConnection,
    ) -> None:
        deadline = time.monotonic() + ICE_GATHER_TIMEOUT

        while pc.iceGatheringState != "complete":
            if time.monotonic() >= deadline:
                raise TimeoutError("ICE gathering timed out")
            await asyncio.sleep(0.1)

    async def _set_session_error(
        self,
        uid: str,
        session_id: str,
        error: str,
    ) -> None:
        try:
            def publish_error(session):
                if (not isinstance(session, dict) or session.get("state") not in (None, "offer", "answering")
                        or not self._session_for_this_pi(session)):
                    return session
                return {**session, "state": "error", "error": error[:500]}
            await self._remote(self.root_ref.child(uid).child(session_id).transaction, publish_error)
        except Exception:
            LOG.exception(
                "Could not publish WebRTC error for %s/%s",
                uid,
                session_id,
            )

    async def _delete_session(
        self,
        uid: str,
        session_id: str,
    ) -> None:
        try:
            def delete_expired(session):
                if isinstance(session, dict) and self._session_for_this_pi(session) and self._session_expired(session):
                    return None
                return session
            await self._remote(self.root_ref.child(uid).child(session_id).transaction, delete_expired)
        except Exception:
            LOG.exception(
                "Could not delete WebRTC session %s/%s",
                uid,
                session_id,
            )

    async def _cleanup_connection(
        self,
        uid: str,
        session_id: str,
    ) -> None:
        key = f"{uid}/{session_id}"
        pc = self.peer_connections.pop(key, None)
        getattr(self, "_peer_deadlines", {}).pop(key, None)

        if pc is not None:
            try:
                await pc.close()
            except Exception:
                LOG.exception(
                    "Could not close WebRTC peer %s/%s",
                    uid,
                    session_id,
                )

    async def _close_all(self) -> None:
        connections = list(self.peer_connections.items())
        self.peer_connections.clear()
        self._peer_deadlines.clear()

        for _, pc in connections:
            try:
                await pc.close()
            except Exception:
                LOG.exception("Could not close WebRTC peer during shutdown")
