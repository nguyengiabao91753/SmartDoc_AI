# Hướng dẫn thiết lập Neo4j Aura cho SmartDoc AI

Chào bạn! Đây là hướng dẫn giúp bạn tự tạo một cơ sở dữ liệu đồ thị trên đám mây để kết nối với dự án **SmartDoc AI**.

---

## Bước 1: Đăng ký tài khoản
1. Truy cập: [Neo4j Aura Console](https://console.neo4j.io/)
2. Đăng ký bằng tài khoản Google cho nhanh, hoặc dùng Email cá nhân.

## Bước 2: Tạo Instance (Cơ sở dữ liệu)
1. Trong màn hình Dashboard, nhấn nút **New Instance** hoặc **Create Instance**.
2. Chọn gói **Aura Free** (Gói miễn phí 100%). 
3. Chọn vùng (Region) là **Singapore** (hoặc nơi gần Việt Nam nhất).
4. Nhấn **Create**.

## Bước 3: Lưu trữ thông tin kết nối (QUAN TRỌNG)
Khi instance được tạo xong, một hộp thoại chứa thông tin đăng nhập sẽ hiện ra.
1. Nhấn nút **Download Credentials (.txt)** để tải file về máy.
2. **Lưu file này thật kỹ**, vì **mật khẩu (Password)** này sẽ không bao giờ hiển thị lại lần thứ hai.

## Bước 4: Cấu hình vào Project
Bây giờ, bạn mở mã nguồn project SmartDoc AI trên máy tính của mình và làm theo các bước:

1. Tìm đến thư mục theo đúng đường dẫn này: 
   `backend/app/rag/modes/graphrag/`
2. Tạo một file mới tên là `.env` (Lưu ý: chỉ là `.env`, không có chữ gì đằng trước dấu chấm). Nếu file đã có thì mở nó lên.
3. Mở file `.txt` bạn đã tải ở Bước 3, copy 3 thông tin quan trọng và dán vào file `.env` theo đúng định dạng sau:

```env
NEO4J_URI=neo4j+s://<dãy-kí-tự-ngẫu-nhiên>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<mật-khẩu-của-bạn>
