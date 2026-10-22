# README Basic — How to Run

## 1. Yêu cầu môi trường

Cài đặt các công cụ sau trước khi chạy project:

- **Python:** phiên bản tương thích với các thư viện trong project.
- **Git:** dùng để tải project từ GitHub (nếu cần).
- **PyCharm** hoặc **Visual Studio Code (VS Code)**.
- Nếu project có frontend chạy bằng Node.js, cần cài thêm **Node.js và npm**.

## 2. Tải project về máy

Nếu project đã được đăng lên GitHub:

1. Mở repository của project.
2. Chọn **Code → Download ZIP**.
3. Giải nén file ZIP vào thư mục muốn lưu trên máy.

Hoặc sử dụng Git:

```bash
git clone <GITHUB_REPOSITORY_URL>
```

Thay `<GITHUB_REPOSITORY_URL>` bằng đường dẫn repository thực tế.

## 3. Mở project bằng PyCharm

1. Mở PyCharm.
2. Chọn **Open**.
3. Chọn thư mục gốc của project vừa tải về.
4. Chờ PyCharm tải và nhận diện các file trong project.
5. Mở **Settings → Project → Python Interpreter**.
6. Chọn hoặc tạo môi trường Python ảo cho project.

Mở Terminal trong PyCharm và chuyển đến thư mục gốc của project nếu cần.

## 4. Mở project bằng Visual Studio Code

1. Mở VS Code.
2. Chọn **File → Open Folder**.
3. Chọn thư mục gốc của project.
4. Cài extension **Python** của Microsoft nếu chưa có.
5. Nhấn `Ctrl + Shift + P`.
6. Tìm `Python: Select Interpreter`.
7. Chọn Python interpreter của project hoặc môi trường ảo đã tạo.

Mở Terminal bằng **Terminal → New Terminal**.

## 5. Tạo môi trường Python ảo

Tại Terminal, chạy:

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo trên Windows.

**PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Command Prompt (CMD):**

```bat
.venv\Scripts\activate.bat
```

Sau khi kích hoạt thành công, Terminal thường hiển thị `(.venv)` ở đầu dòng lệnh.

Nếu PowerShell không cho phép kích hoạt, có thể dùng CMD hoặc chọn interpreter của `.venv` trực tiếp trong IDE.

## 6. Cài đặt thư viện

Nếu project có file `requirements.txt`, chạy:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu chưa có `requirements.txt`, hãy cài các thư viện được khai báo trong tài liệu hoặc cấu hình thực tế của project. Không nên tự cài các phiên bản ngẫu nhiên vì có thể gây xung đột.

## 7. Kiểm tra các file cần thiết

Trước khi chạy, kiểm tra project đã có những thành phần cần thiết:

- File dữ liệu gốc trong thư mục `data/`, nếu chức năng cần đọc dữ liệu.
- Các mô hình đã huấn luyện trong thư mục `models/`, nếu ứng dụng tải mô hình có sẵn.
- Các file cấu hình cần thiết.
- File `.env` nếu project yêu cầu biến môi trường.

Nếu thiếu file mô hình hoặc dữ liệu, hãy hỏi người quản lý project để nhận đúng file. Không tự tạo file thay thế bằng dữ liệu giả.

## 8. Chạy ứng dụng

### Trường hợp A: Frontend sử dụng Streamlit

Nếu project có file `app.py` là ứng dụng Streamlit, chạy tại thư mục chứa file:

```bash
streamlit run app.py
```

Sau đó mở địa chỉ Local URL được hiển thị trong Terminal, thường là:

```text
http://localhost:8501
```

### Trường hợp B: Frontend sử dụng React/Vite

Nếu project có thư mục frontend chứa `package.json`:

```bash
cd <FRONTEND_FOLDER>
npm install
npm run dev
```

Mở địa chỉ được hiển thị trong Terminal.

Nếu project yêu cầu backend riêng, hãy chạy backend theo hướng dẫn và cấu hình thực tế trước hoặc đồng thời với frontend.

### Trường hợp C: Chỉ chạy chương trình Python

Nếu project có script Python độc lập, chạy file entry point được chỉ định trong project, ví dụ:

```bash
python <ENTRY_POINT>.py
```

Thay `<ENTRY_POINT>.py` bằng tên file chạy thực tế.

## 9. Một số lỗi thường gặp

- **ModuleNotFoundError:** Kiểm tra đã kích hoạt đúng môi trường ảo và cài các thư viện cần thiết chưa.
- **FileNotFoundError:** Kiểm tra đường dẫn dữ liệu, mô hình và thư mục làm việc.
- **Không tìm thấy mô hình:** Kiểm tra các file mô hình đã được cung cấp đầy đủ chưa.
- **Frontend không kết nối được backend:** Kiểm tra backend đã chạy chưa và địa chỉ API có đúng không.
- **Port đã được sử dụng:** Kiểm tra ứng dụng khác có đang sử dụng cổng đó không.

## 10. Lưu ý

- Chạy lệnh tại đúng thư mục được hướng dẫn.
- Không đưa thư mục `.venv/`, cache hoặc file chứa thông tin bí mật lên GitHub.
- Không tự ý đổi tên hoặc di chuyển các file dữ liệu, mô hình và cấu hình.
- Nếu gặp lỗi, hãy gửi ảnh Terminal hoặc sao chép toàn bộ thông báo lỗi để nhóm hỗ trợ.

**Hoàn thành:** Khi ứng dụng khởi động thành công và giao diện mở được trên trình duyệt, việc cài đặt và chạy project đã hoàn tất.