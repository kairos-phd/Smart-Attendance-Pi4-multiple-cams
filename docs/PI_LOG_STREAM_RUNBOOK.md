# Console log, Recog và stream Pi

## Kết quả đối chiếu log ngày 10/10/2026

- Bản `52a5436` đã chạy **367 test trên Pi**, và ONNX preflight đọc được output INT8 scale `0.02524387650191784`, zero point `0`.
- Cập nhật dừng vì **2 active user không khớp fingerprint mới**, nên dịch vụ cũ được giữ nguyên. `/health` chưa có `model_info` là đúng với source cũ; không phải cập nhật thành công rồi mất metadata.
- Runtime cũ có `recognition.rejected=1764`; feed `/api/recognitions` lọc `accepted`, nên console có thể trống dù inference đang chạy. Đây là lỗi phạm vi feed, không phải bằng chứng SQLite mất dữ liệu. Ba file text không có traceback cụ thể khi bấm Recog.
- Thời điểm lấy `/health`: `recognition_enabled=0`, ba camera `icspring`, `jieli`, `sonix` capture khoảng 4 FPS. CPU/RAM lúc đó không dùng làm benchmark khi Recog bật. Chưa có ICE/browser stats để kết luận WebRTC đã stream thành công.

## Xem log ngay trên source Pi cũ

Lệnh dưới chỉ lấy watcher từ GitHub và đọc log/database hiện có. Không cập nhật runtime, không stop service, không đổi fingerprint/embedding:

```bash
cd /home/sic4b/Smart-Attendance-Pi4-multiple-cams
git fetch origin refs/heads/fix/recognition-event-logs:refs/remotes/origin/fix/recognition-event-logs
git show origin/fix/recognition-event-logs:scripts/watch_recognition.py > /tmp/sic-watch-recognition.py

# Xem 20 sự kiện gần nhất và trạng thái hiện tại:
python3 /tmp/sic-watch-recognition.py --repo "$PWD" --once

# Mở ở terminal riêng rồi bấm Recog trong console; Ctrl+C để dừng watcher.
umask 077
python3 -u /tmp/sic-watch-recognition.py --repo "$PWD" --limit 20 \
  | tee "/tmp/sic-recognition-$(date +%Y%m%d-%H%M%S).jsonl"
```

JSONL dùng event ID của recognition trong SQLite, giữ camera, timestamp, similarity, reason và model. `recognition_state` ghi enabled, model, previous model và `model_changed`. Bật/tắt Recog chỉ đổi cờ xử lý, không load/chuyển model. Poll có thể không thấy một toggle diễn ra hoàn toàn giữa hai lần đọc; journal của source mới ghi từng API toggle.

`unknown` nghĩa là có face observation nhưng không có candidate hoặc similarity chưa đạt; quality/detection/margin thấp là `rejected` với reason rõ ràng. Không phát hiện face thì không tạo Unknown/attendance giả. Unknown không được tính là attendance. User bị reject không hiện tên candidate như đã nhận diện thành công.

Nếu DB bị restore/xóa và rowid lùi, watcher ghi `log_cursor_reset` rồi có thể phát lại các event cũ; đối chiếu `observation_id` khi tổng hợp JSONL. Việc này không tạo bản ghi attendance mới.

Watcher đọc SQLite mode `ro`, không migrate schema và không chuyển sự kiện lên cloud. Cần user Linux có quyền đọc DATA_ROOT/database/WAL. `.env` chỉ đọc DATA_ROOT/API_PORT; không in token/credential. Nếu service EnvironmentFile khác repo `.env`, dùng `--data-root` và `--url http://127.0.0.1:PORT` khớp service. Report chứa thông tin điểm danh, giữ riêng khi chia sẻ.

Nếu `diagnostic_error` xuất hiện, giữ nguyên dòng lỗi. `HTTP 401`/`403` ở console API thông thường cần kiểm tra auth mode/token; không đổi endpoint sang public để bỏ lỗi. Watcher dùng localhost control cùng database để hỗ trợ chẩn đoán runtime cũ.

## Kiểm tra stream và inference khi bật Recog

Mở ba terminal SSH/console. Terminal thứ nhất:

```bash
sudo journalctl -u sic-pi4-attendance --since '10 minutes ago' \
  --no-pager -l -o short-iso \
  | rg -i 'recognition|inference|webrtc|ice|traceback|error|failed'

sudo journalctl -u sic-pi4-attendance -f -l -o short-iso
```

Terminal thứ hai chạy watcher ở trên. Terminal thứ ba lưu trạng thái trước/sau khi bật Recog:

```bash
umask 077
curl -fsS http://127.0.0.1:8080/health > /tmp/sic-health-recog-off.json
curl -fsS http://127.0.0.1:8080/api/control | python3 -m json.tool

# Bấm Recog trên console, cho người có trong dữ liệu và người lạ đi qua từng camera.
# Chờ đủ thời gian lấy mẫu, sau đó lưu:
curl -fsS http://127.0.0.1:8080/health > /tmp/sic-health-recog-on.json
curl -fsS http://127.0.0.1:8080/api/control | python3 -m json.tool
vcgencmd measure_temp
vcgencmd get_throttled
```

So sánh captured/enqueued/processed rates, recognition enabled, inference p95/queue p95 và CPU. `camera_health.last_seen` không thay thế kiểm tra frame/capture mới. Khung video lấy từ preview, không phải gọi inference mỗi lần browser nhận frame. Recog tắt thì video vẫn chạy, Recog bật thì video có thể có overlay; enrollment tiếp tục dùng ảnh gốc.

Source đã kiểm thử raw → overlay → raw và RTP timestamp tăng ở cả ba camera khi bật/tắt Recog. Đây là test logic, chưa chứng minh USB/CPU/ICE/NAT và browser thực tế. Ở browser cần kiểm tra video có frame mới và WebRTC `getStats()` có inbound frames/packets tăng; nếu chỉ kết nối ICE mà không có frame thì chưa đạt. STUN có thể không đủ qua mạng khác/NAT; TURN cần cấu hình riêng và kiểm thử tải.

## Source mới và web

- Console mặc định dùng `/api/recognition-events`; `--accepted-only` giữ feed cũ nếu cần đối chiếu.
- `/api/recognitions` giữ contract accepted-only cũ, `/api/attendance` vẫn là attendance đã ghi.
- `/api/recognition-events` cung cấp `observation_id`, `display_status`, `status`, `reason`, `model_version` và các trường camera/time/score.
- `/api/control/remote` GET/PATCH yêu cầu admin bearer auth; `/api/control` nguyên gốc vẫn chỉ cho loopback. PATCH `recognition_enabled` áp dụng toàn bộ camera; `camera_id` chỉ chọn camera hiển thị.
- Token quản trị của Pi chỉ đặt ở backend server, không nhúng trong JS/Vite hoặc URL.

Bản sửa hoàn chỉnh còn dùng fingerprint mới của nhánh trước. **Chưa áp dụng full updater lên Pi có 2 embedding cũ** cho đến khi xác minh lịch sử model/ảnh enrollment hoặc có quy trình tạo lại embedding được kiểm tra. Không dùng SQL đổi `model_version`, không bật `--allow-incompatible` cho vận hành điểm danh chính thức. Watcher đã giải quyết nhu cầu xem log ngay mà không cần bypass gate.

Sau khi catalog đã tương thích, cập nhật bằng updater của nhánh mới, theo `docs/PI_UPDATE_AND_FILE_AUDIT.md`, thay ref thành `origin/fix/recognition-event-logs`.

Website có panel Pi riêng và backend proxy trong repo `Smart-Attendance-Pi4-Webserver` nhánh `fix/pi-stream-control`. Xem runbook đi kèm nhánh đó để cấu hình API luôn bật, Pi tunnel, Firebase RTDB và viewer quyền admin. Chưa triển khai server/tunnel/rules/website tự động.

## Bước tiếp theo theo lộ trình

1. Lưu watcher log, health khi Recog on/off, journal lỗi và preflight. Xác minh model cũ/mới với ảnh enrollment đã có nhãn; giải quyết 2 user trước rollout đầy đủ. Giữ mọi model reference, DB và ảnh.
2. Test 1/2/3 camera với người đã enroll và người lạ, đo CPU p95, inference/queue/end-to-end p95, nhiệt độ/throttling và FAR/FRR. Thử đồng thời 1/2/3 viewers để tách tải inference khỏi encoding.
3. Chạy API/stream thử nghiệm, cắt mạng rồi gửi bù; kiểm tra idempotency/event ID và lỗi network không ngăn điểm danh local.
4. Đưa media qua đường riêng tới sic4b.io.vn và khóa viewer/control. Chỉ bỏ model webserver sau khi Pi đạt tiêu chí benchmark; host frontend tĩnh không thay thế API đang nằm trên PC.
5. Thu ảnh/manifest có consent và duyệt nhãn; PAD và thử tại ga theo các tiêu chí/phương án lỗi/quyền riêng tư đã thống nhất.
