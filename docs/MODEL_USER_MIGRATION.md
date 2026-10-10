# Giữ registered users khi thay model

Có thể giữ user ID, student ID và lịch sử điểm danh bằng cách tạo lại embeddings từ ảnh enrollment gốc đã duyệt nhãn. Đổi recognizer hoặc quantize FP32 → INT8 cần chạy ảnh qua recognizer mới; không đổi nhãn `model_version` trong SQL để ép embeddings cũ khớp. Thay detector cũng cần kiểm tra alignment/quality. Chỉ đổi định dạng lưu embedding của cùng model mới có thể chuyển dequantize/requantize mà không inference lại, vẫn cần đánh giá sai số.

Hiện source chưa có công cụ migration user ID tự động: enrollment tạo user mới; import catalog không phải update embeddings tại chỗ. Media-only updater không đổi model và không làm migration. Giữ recognizer đang nhận diện được cho đến khi hoàn thành các bước sau:

1. Kiểm kê ảnh enrollment tồn tại, SHA256 đúng và nhãn đã duyệt. User thiếu ảnh phải thu lại ảnh; không tự đoán danh tính.
2. Dùng model manifest/fingerprint mới và pipeline enrollment để tạo catalog staging. Giữ `user_id`/`student_id`; không insert người trùng.
3. Đo nhiều người, unknown, nhiều camera, ánh sáng/khoảng cách; calibrate threshold mới bằng dữ liệu có ground truth để đo FAR/FRR. Chỉ đo CPU không đủ chấp nhận model.
4. Dừng service ngắn; backup SQLite nhất quán và model/config. Trong transaction, chỉ cập nhật embedding, dimension, scale, model version và timestamp/audit; giữ users, ảnh và lịch sử attendance. Kiểm tra catalog đúng fingerprint trước restart.
5. Nếu không đạt: phục hồi đồng bộ DB + model + config rồi restart. Không chỉ rollback một phía.

Kiểm kê đọc-only trên Pi (chỉ in số lượng/trạng thái ảnh, không in tên hay nội dung):

```bash
python3 - <<'PY'
from pathlib import Path
import sqlite3
root = Path('/var/lib/sic-attendance')
database = root / 'smart_attendance.sqlite3'
with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as db:
    print('users:', db.execute('SELECT COUNT(*) FROM users').fetchone()[0])
    print('image schema:', [r[1] for r in db.execute('PRAGMA table_info(enrollment_images)')])
    print('images:', db.execute('SELECT COUNT(*) FROM enrollment_images').fetchone()[0])
    print('model groups:', db.execute('SELECT model_version,COUNT(*) FROM users GROUP BY model_version').fetchall())
PY
```

Đọc schema thực tế trước khi viết công cụ migration; schema/ảnh trên Pi không được suy ra từ source archive. Bước tiếp theo sau stream là soak 30 phút với model hiện tại, thu ground truth, rồi mới so sánh INT8/model mới. Tiếp theo mới kiểm thử API mất mạng/gửi bù idempotent, công cụ duyệt nhãn và PAD. Thử tại ga sau khi có tiêu chí tải, độ chính xác, xử lý lỗi và quyền riêng tư.
