import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from pi4_attendance.api import create_app
from pi4_attendance.preview import PreviewWindow
from pi4_attendance.service import AttendanceRuntime
from pi4_attendance.tools.console import request_json


def control_runtime(on_enroll=None) -> AttendanceRuntime:
    runtime = AttendanceRuntime.__new__(AttendanceRuntime)
    runtime.settings = SimpleNamespace(
        api_cors_origins=[], api_auth_mode="shared_token", admin_shared_token="secret"
    )
    runtime.preview = PreviewWindow(("camera-2", "camera-1"), 30.0, on_enroll)
    runtime.recognition_enabled = runtime.preview.recognition_enabled
    runtime._runtime_state = "running"
    runtime._camera_cloud_lock = threading.Lock()
    runtime._online_camera_ids = {"camera-2", "camera-1"}
    return runtime


def test_loopback_control_status_and_updates_use_shared_runtime_state() -> None:
    runtime = control_runtime()

    async def exercise() -> None:
        transport = httpx.ASGITransport(app=create_app(runtime), client=("127.0.0.1", 41000))
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            response = await client.get("/api/control")
            assert response.status_code == 200
            assert response.json() == {
                "camera_ids": ["camera-2", "camera-1"],
                "selected_camera_id": "camera-2",
                "recognition_enabled": False,
                "enrollment": None,
                "enrollment_busy": False,
                "enrollment_status": "",
                "runtime_state": "running",
                "online_camera_ids": ["camera-1", "camera-2"],
            }

            response = await client.patch(
                "/api/control",
                json={"camera_id": "camera-1", "recognition_enabled": True},
            )
            assert response.status_code == 200
            assert response.json()["selected_camera_id"] == "camera-1"
            assert response.json()["recognition_enabled"] is True

            response = await client.patch("/api/control", json={"recognition_enabled": True})
            assert response.status_code == 200
            assert response.json()["recognition_enabled"] is True

            response = await client.patch("/api/control", json={"camera_id": "missing"})
            assert response.status_code == 400

    asyncio.run(exercise())


def test_loopback_enrollment_routes_and_status_codes() -> None:
    enrolled = []
    enrolled_event = threading.Event()

    def enroll(name, student_id, images, actor):
        enrolled.append((name, student_id, len(images), actor))
        enrolled_event.set()
        return {"status": "active"}

    runtime = control_runtime(enroll)

    async def exercise() -> None:
        transport = httpx.ASGITransport(app=create_app(runtime), client=("::1", 41000))
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            response = await client.post(
                "/api/control/enrollment/start",
                json={"name": " Dang ", "student_id": " S001 "},
            )
            assert response.status_code == 200
            assert response.json()["enrollment"] == {
                "name": "Dang",
                "student_id": "S001",
                "captured": 0,
                "actor": "local-console",
            }

            response = await client.post(
                "/api/control/enrollment/start", json={"name": "Other", "student_id": "S002"}
            )
            assert response.status_code == 409
            assert (await client.post("/api/control/enrollment/capture")).status_code == 400
            assert (await client.post("/api/control/enrollment/submit")).status_code == 400

            for value in range(3):
                runtime.preview.submit("camera-2", np.full((2, 2, 3), value, dtype=np.uint8))
                response = await client.post("/api/control/enrollment/capture")
                assert response.status_code == 200
                assert response.json()["enrollment"]["captured"] == value + 1

            assert (await client.post("/api/control/enrollment/submit")).status_code == 200
            assert await asyncio.to_thread(enrolled_event.wait, 1)
            assert enrolled == [("Dang", "S001", 3, "local-console")]

            for _ in range(100):
                if (await client.get("/api/control")).json()["enrollment"] is None:
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("Enrollment worker did not finish")

            response = await client.post(
                "/api/control/enrollment/start", json={"name": "Cancel", "student_id": "S003"}
            )
            assert response.status_code == 200
            assert (await client.delete("/api/control/enrollment")).status_code == 200
            assert (await client.delete("/api/control/enrollment")).status_code == 400

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "", "student_id": "S001"},
        {"name": "   ", "student_id": "S001"},
        {"name": "x" * 121, "student_id": "S001"},
        {"name": "Dang", "student_id": ""},
        {"name": "Dang", "student_id": "x" * 81},
    ],
)
def test_enrollment_request_model_returns_400_for_invalid_input(payload: dict) -> None:
    async def exercise() -> None:
        transport = httpx.ASGITransport(
            app=create_app(control_runtime()), client=("127.0.0.1", 41000)
        )
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            response = await client.post("/api/control/enrollment/start", json=payload)
            assert response.status_code == 400

    asyncio.run(exercise())


def test_all_control_routes_deny_non_loopback_clients() -> None:
    runtime = control_runtime()

    async def exercise() -> None:
        transport = httpx.ASGITransport(app=create_app(runtime), client=("192.0.2.10", 41000))
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            requests = (
                client.get("/api/control"),
                client.patch("/api/control", json={"recognition_enabled": True}),
                client.post(
                    "/api/control/enrollment/start", json={"name": "Dang", "student_id": "S001"}
                ),
                client.post("/api/control/enrollment/capture"),
                client.post("/api/control/enrollment/submit"),
                client.delete("/api/control/enrollment"),
            )
            responses = await asyncio.gather(*requests)
            assert [response.status_code for response in responses] == [403] * 6
            assert runtime.preview.status()["recognition_enabled"] is False
            assert runtime.preview.status()["enrollment"] is None

    asyncio.run(exercise())


@pytest.mark.parametrize("address", [None, ("not-an-ip", 41000)])
def test_control_routes_deny_missing_or_malformed_client_addresses(address) -> None:
    async def exercise() -> None:
        transport = httpx.ASGITransport(app=create_app(control_runtime()), client=address)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            assert (await client.get("/api/control")).status_code == 403

    asyncio.run(exercise())


@pytest.mark.parametrize("headers", [
    [("host", value)] for value in (
        "attacker.example", "", "localhost/path", "localhost?", "localhost#", "user@localhost",
        "127.0.0.1:", "[::1]:", "localhost:bad", "localhost:65536", "localhost:-1",
        " localhost", "local\thost", "localhost\n", "::1", "[::1%eth0]", "[127.0.0.1]",
    )
] + [[], [("host", "localhost"), ("host", "127.0.0.1")]] + [
    [("host", "127.0.0.1"), ("origin", value)]
    for value in ("", "null", "http://127.0.0.1", "https://attacker.example")
])
def test_control_boundary_rejects_all_routes_without_mutation(headers) -> None:
    enrolled = []
    runtime = control_runtime(lambda *args: enrolled.append(args))
    runtime.begin_enrollment("Dang", "S001", "local-console")
    for value in range(3):
        runtime.preview.submit("camera-2", np.full((2, 2, 3), value, dtype=np.uint8))
        runtime.capture_enrollment_frame()
    runtime.preview.submit("camera-2", np.full((2, 2, 3), 4, dtype=np.uint8))
    before = runtime.control_status()

    async def exercise():
        transport = httpx.ASGITransport(app=create_app(runtime), client=("127.0.0.1", 41000))
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            for method, path, payload in (
                ("GET", "/api/control", None),
                ("PATCH", "/api/control", {"camera_id": "camera-1", "recognition_enabled": True}),
                ("POST", "/api/control/enrollment/start", {"name": "Other", "student_id": "S002"}),
                ("POST", "/api/control/enrollment/capture", None),
                ("POST", "/api/control/enrollment/submit", None),
                ("DELETE", "/api/control/enrollment", None),
            ):
                request = client.build_request(method, path, json=payload)
                request.headers = httpx.Headers(headers)
                if payload is None and method == "POST":
                    request.headers["content-type"] = "application/x-www-form-urlencoded"
                response = await client.send(request)
                assert response.status_code == 403, (path, response.text)
                assert runtime.control_status() == before
                assert enrolled == []
    asyncio.run(exercise())


@pytest.mark.parametrize("peer,host", [
    ("127.0.0.1", "localhost"), ("127.0.0.2", "LOCALHOST:8080"),
    ("::1", "[::1]"), ("::1", "[::1]:8080"), ("127.0.0.1", "127.0.0.2:80"),
])
def test_control_boundary_accepts_loopback_authorities(peer, host) -> None:
    async def exercise():
        transport = httpx.ASGITransport(app=create_app(control_runtime()), client=(peer, 41000))
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            response = await client.get("/api/control", headers={"host": host})
            assert response.status_code == 200
    asyncio.run(exercise())


def test_stdlib_client_reaches_control_route_with_loopback_host_and_no_origin() -> None:
    runtime = control_runtime()
    runtime.begin_enrollment("Dang", "S001", "local-console")
    runtime.preview.submit("camera-2", np.zeros((2, 2, 3), dtype=np.uint8))
    app = create_app(runtime)
    observed = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            observed.append(dict(self.headers))

            async def dispatch():
                transport = httpx.ASGITransport(app=app, client=self.client_address)
                async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
                    return await client.post(self.path, headers=list(self.headers.items()))

            response = asyncio.run(dispatch())
            self.send_response(response.status_code)
            self.end_headers()
            self.wfile.write(response.content)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        result = request_json(f"http://127.0.0.1:{server.server_port}", "/api/control/enrollment/capture", method="POST")
        assert result["enrollment"]["captured"] == 1
        assert observed[0]["Host"] == f"127.0.0.1:{server.server_port}"
        assert "origin" not in {name.lower() for name in observed[0]}
    finally:
        server.shutdown()
        server.server_close()
        worker.join(1)
