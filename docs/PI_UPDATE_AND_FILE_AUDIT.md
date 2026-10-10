# Cập nhật Pi hiện tại và kiểm kê từng file

Áp dụng cho source và report ngày 07/10/2026 đã cung cấp: Pi `192.168.0.208`, user `sic4b`, repo `/home/sic4b/Smart-Attendance-Pi4-multiple-cams`. Bản sửa giữ các thay đổi Pi đã gửi, gồm preview/WebRTC, camera, Firebase/API; không thay bằng source GitHub cũ.

## Những lỗi được sửa

- Thiếu dependency media không còn ngăn API khởi động. Khai báo `aiortc`, `av`; cho phép Firebase Admin 7.x đang cài trên Pi.
- Nhận diện, enrollment, benchmark và export/import embedding dùng cùng loader và fingerprint model. Fingerprint bao gồm detector/recognizer thực tế và preprocessing; đọc output scale/zero point từ ONNX thay vì hardcode. Hỗ trợ output float, int8 và uint8 với hợp đồng đầu ra được kiểm tra.
- MJPEG kiểm tra lại phiên trong lúc stream. Cookie có Secure mặc định; cập nhật Pi phải chọn HTTP hoặc HTTPS rõ ràng.
- WebRTC dùng transaction để nhận offer và công bố answer, không ghi đè phiên đã bị thu hồi/xóa. Watchdog đóng peer hết hạn độc lập với Firebase; giới hạn số người xem đồng thời và thời gian chờ signaling.
- Khung enrollment giữ ảnh gốc. HDMI/MJPEG/WebRTC hiển thị overlay riêng; bỏ một lần copy dư cho mỗi khung camera.
- Firestore summary không khởi tạo client RTDB không dùng. `FIREBASE_DATABASE_URL` có default trong Settings để không phá các call site cũ.
- `/health` công bố `model_info.files` cùng thông tin hợp đồng preprocessing để biết file thật đang chạy.

## Cập nhật từ GitHub

Trên Windows PowerShell, kết nối Pi:

```powershell
ssh sic4b@192.168.0.208
```

Các lệnh tiếp theo chạy **trên Pi**. Không chạy `git pull`, `git clean` hoặc `git reset --hard` trên checkout có thay đổi chưa lưu.

```bash
cd /home/sic4b/Smart-Attendance-Pi4-multiple-cams

git fetch origin refs/heads/fix/pi-runtime-and-file-audit:refs/remotes/origin/fix/pi-runtime-and-file-audit
git log -1 --oneline origin/fix/pi-runtime-and-file-audit

git show origin/fix/pi-runtime-and-file-audit:scripts/update_pi.py > /tmp/sic-update-pi.py

# Chỉ cài dependency còn thiếu; lệnh không ép nâng cấp package đã đáp ứng range.
.venv/bin/python -m pip install 'aiortc>=1.14,<2' 'av>=16,<18' 'firebase-admin>=6.5,<8' 'pytest>=8,<9' 'httpx>=0.27,<1' 'setuptools>=68' wheel

# Kiểm tra; chưa chuyển source, chưa stop/restart service.
python3 /tmp/sic-update-pi.py --repo "$PWD" --ref origin/fix/pi-runtime-and-file-audit
```

Script in đường dẫn `/home/sic4b/sic-backups/<timestamp>/model-preflight.json`. Đọc JSON đó để biết model thật, SHA256, output contract và số user tương thích. Script có thể cần thời gian load model khi dịch vụ cũ vẫn đang hoạt động.

Nếu preflight đạt, cập nhật LAN HTTP hiện có:

```bash
python3 /tmp/sic-update-pi.py --repo "$PWD" --ref origin/fix/pi-runtime-and-file-audit --apply --preview-transport http
sudo systemctl status sic-pi4-attendance --no-pager
curl -fsS http://127.0.0.1:8080/health | python3 -m json.tool
sudo journalctl -u sic-pi4-attendance -n 80 --no-pager
```

Nếu preview qua HTTPS, thay `--preview-transport http` bằng `--preview-transport https`. HTTP dùng cookie `Secure=false`; chỉ dùng trên LAN tin cậy. HTTPS dùng `Secure=true`. Script không thay tài khoản preview, token quản trị hoặc cấu hình camera.

Script tạo nhánh local `pi-deploy/<timestamp>` tại commit đã kiểm tra. Source sửa cục bộ được giữ trong Git stash và archive riêng; file chưa track trùng đường dẫn incoming được chuyển sang backup, các file chưa track khác vẫn ở nguyên chỗ. `.env` giữ nội dung cũ, chỉ cập nhật Secure-cookie và pin recognizer mà preflight đã xác nhận nếu trước đó chọn `auto`. Model và dữ liệu gốc không bị sửa/xóa. Sau khi stop service, catalog được kiểm tra lại để bắt enrollment phát sinh trong lúc preflight; SQLite được backup bằng API backup để nhất quán với WAL. Nếu activation lỗi, script chuyển về source/config cũ, phục hồi stash và những file đã chuyển; thư mục backup vẫn còn để kiểm tra. Cần quyền đọc database/model và quyền `sudo systemctl`. Script xác nhận WorkingDirectory và EnvironmentFile của service khớp repo; môi trường SSH không được dùng để thay DATA_ROOT/model selector của service. Editable install dùng package build đã có, không tải dependency lúc activation/rollback.

## Khi embedding cũ không tương thích

Source Pi cũ ghi fingerprint của pack dù thực tế chạy recognizer `/opt/sic-attendance/models/w600k_mbf_qdq_int8.onnx` ngoài pack. Không đủ chứng cứ để đổi nhãn tất cả embedding cũ sang fingerprint mới. Pack mode cũng tạo fingerprint mới để tránh nhận nhầm dữ liệu từ recognizer ngoài pack.

`incompatible_users > 0` khiến cập nhật dừng **trước khi đổi source hoặc stop service**. Không sửa trực tiếp `users.model_version` bằng SQL. Giữ database, ảnh enrollment và các model tham chiếu. Cần xác nhận lịch sử model, đối chiếu bằng ảnh có nhãn hoặc xây lại embedding từ ảnh gốc đã được duyệt.

Nếu chủ động chấp nhận chạy thử với các user không tương thích chưa thể nhận diện, có tùy chọn `--allow-incompatible` cùng `--apply`. Tùy chọn này không đổi nhãn, không xóa user và không tự migrate. Việc reenroll user hiện có cần quy trình có kiểm soát vì student ID là duy nhất. Không dùng override cho vận hành điểm danh chính thức.

Nếu actual ONNX không đáp ứng hợp đồng đã kiểm tra, preflight cũng dừng. Model thật chưa có trong archive đã gửi; cần xem lỗi và graph/model manifest trước khi quyết định export lại, không đoán scale/zero point.

## Kiểm kê từng file, không xóa

Tool độc lập chỉ dùng thư viện chuẩn Python, không cần activate venv. Sau khi đã cập nhật:

```bash
cd /home/sic4b/Smart-Attendance-Pi4-multiple-cams
python3 scripts/check_pi_files.py --health-url http://127.0.0.1:8080/health --output /tmp/sic-file-audit.json > /tmp/sic-file-audit.tsv
less /tmp/sic-file-audit.tsv
```

Nếu cập nhật bị gate chặn, có thể kiểm kê bằng tool từ nhánh GitHub mà không đổi service:

```bash
git show origin/fix/pi-runtime-and-file-audit:scripts/check_pi_files.py > /tmp/sic-check-pi-files.py
python3 /tmp/sic-check-pi-files.py --repo "$PWD" --health-url http://127.0.0.1:8080/health --output /tmp/sic-file-audit.json > /tmp/sic-file-audit.tsv
```

Runtime cũ chưa có `model_info.files` thì tool giữ tất cả ONNX trong pack cấu hình và recognizer ngoài pack một cách thận trọng. Quyết định `KEEP_ACTIVE_MODEL` khi đó dựa vào config, không chứng minh process đang sử dụng file. Tool ghi rõ nguồn chứng cứ. File model khác luôn là `REVIEW_MODEL`; không bao giờ tự đề xuất xóa model.

Kiểm tra riêng từng file đáng chú ý trong report, bằng tool ở `/tmp` (dùng được trước hoặc sau cập nhật):

```bash
cd /home/sic4b/Smart-Attendance-Pi4-multiple-cams
git show origin/fix/pi-runtime-and-file-audit:scripts/check_pi_files.py > /tmp/sic-check-pi-files.py

for relative_file in \
  src/pi4_attendance/api.py.bak \
  src/pi4_attendance/api.py.bak2 \
  src/pi4_attendance/auth.py.bak \
  src/pi4_attendance/cloud.py.bak \
  src/pi4_attendance/face_engine.py.bak_before_qdq \
  cloudflared.deb \
  dmxcc
 do
  python3 /tmp/sic-check-pi-files.py --repo "$PWD" --file "$PWD/$relative_file" --hash
 done

# File model đang chạy và model dự phòng:
python3 /tmp/sic-check-pi-files.py --repo "$PWD" --file /opt/sic-attendance/models/w600k_mbf_qdq_int8.onnx --hash
python3 /tmp/sic-check-pi-files.py --repo "$PWD" --file /opt/sic-attendance/models/w600k_mbf_fp32_backup.onnx --hash

# Kiểm tra file lạ/installer, không thực thi chúng:
file -- cloudflared.deb dmxcc
dpkg-query -W -f='${Status}\t${Version}\n' cloudflared
systemctl show sic-pi4-attendance -p FragmentPath -p DropInPaths -p WorkingDirectory -p ExecStart -p EnvironmentFiles
```

Có thể thêm `--extra-root /etc/systemd/system --extra-root /usr/local/bin` để kiểm kê các service/script ngoài repo. `--file` hiển thị metadata, symlink, hash nếu yêu cầu và package owner nếu tìm được; không in nội dung credential. File đã mất sẽ được báo lỗi. Tên/path không rõ thì `KEEP_REVIEW`.

| Quyết định | Xử lý |
| --- | --- |
| `KEEP_PRIVATE_DATA` | Giữ `.env`, credential, database/backup, ảnh và calibration. Không gửi lên GitHub. |
| `KEEP_ACTIVE_MODEL` | Giữ model chọn theo runtime/config, kể cả symlink. |
| `KEEP_RUNTIME`, `KEEP_DEPENDENCY` | Giữ source, deploy script, `.venv`, `.git`. |
| `KEEP_VALIDATION`, `KEEP_REFERENCED` | Giữ docs/tests hoặc file có static reference. |
| `ARCHIVE_CANDIDATE` | Backup code không thấy reference: có thể chuyển sang thư mục backup riêng sau khi kiểm tra, không cần xóa. |
| `REVIEW_PACKAGE` | `.deb`: xác nhận đã cài và không cần phục hồi offline trước khi lưu trữ nơi khác. |
| `REVIEW_MODEL` | Giữ để so sánh/rollback cho đến khi benchmark đạt. |
| `REGENERABLE` | Cache/build metadata: kiểm tra trước; `.egg-info` có thể cần cho metadata/entry point editable. |
| `KEEP_SYSTEM`, `KEEP_REVIEW` | Giữ; kiểm tra unit/package/reference hoặc yêu cầu thêm chứng cứ. |

JSON có quyền `0600`; không coi report là bằng chứng tuyệt đối về mọi import động, open file hoặc process bên ngoài dự án.

Lấy report về Windows ổ F bằng PowerShell:

```powershell
scp sic4b@192.168.0.208:/tmp/sic-file-audit.json F:\sic-file-audit.json
scp sic4b@192.168.0.208:/tmp/sic-file-audit.tsv F:\sic-file-audit.tsv
```

## Optimize theo lộ trình

1. Giữ manifest runtime từ `model-preflight.json` và `/health`, pin recognizer đã xác nhận; không sửa manifest pack cũ để ép hợp thức hóa embedding. Giữ bản FP32 và dữ liệu calibration.
2. Camera dùng `/dev/v4l/by-id` hoặc `by-path` đã xác nhận capture. Report có lỗi V4L2 mở device theo tên: kiểm tra đường dẫn, format và camera health trên Pi. Không thể suy ra camera nào hỏng chỉ từ source.
3. Bắt đầu ở 640×480, `MONITOR_FPS=4`, thử lần lượt 1/2/3 camera. Giữ hàng đợi có giới hạn, catalog embedding đã cache, giới hạn viewers. Chọn FPS/queue theo p95 và dropped frames thực tế; không tăng queue để giấu backlog.
4. Benchmark đúng file đang chạy, cùng detector và bộ ảnh có nhãn đã duyệt:

```bash
.venv/bin/python -m pi4_attendance.tools.benchmark \
  --model-root /opt/sic-attendance \
  --reference pi4_fp32_new --candidate pi4_fp32_new \
  --reference-recognizer /opt/sic-attendance/models/w600k_mbf_fp32_backup.onnx \
  --candidate-recognizer /opt/sic-attendance/models/w600k_mbf_qdq_int8.onnx \
  --dataset /duong-dan/tap-anh-da-duyet \
  --output /tmp/sic-model-benchmark.json
```

Tên pack ở ví dụ phải khớp preflight và có detector hợp lệ. Benchmark tính latency/model comparison và pairwise FAR/FRR. Nó chưa thay thế bài test live nhiều người/nhiều camera/người lạ: cần đo system CPU p95, queue age/depth, nhiệt độ, throttling, độ trễ end-to-end và open-set FAR/FRR trên Pi. Xem `docs/DEPLOYMENT.md` để lấy performance log.

5. Offline/replay phải thử với API đích thật và kiểm tra event ID/idempotency. Test outbox local đã có nhưng source web/API mới và Firebase rules chưa được cung cấp.
6. Stream sang `sic4b.io.vn` cần media path riêng, ACL theo user/camera/Pi, RTDB rules phía client, TURN và cấu hình reverse proxy/tunnel phù hợp. Server Admin SDK bypass rules; transaction/TTL không thay thế quyền viewer trong rules. Không upload rules hoặc mở public stream tự động. Chỉ bỏ inference khỏi webserver sau khi Pi đạt tiêu chí tải/độ chính xác.
7. Thu ảnh có consent, duyệt nhãn/manifest, PAD trên tập tấn công có kiểm soát; chỉ thử ở ga khi có tiêu chí nghiệm thu và phương án lỗi/quyền riêng tư. Chưa có chứng cứ để xác nhận các bước này đã đạt.
