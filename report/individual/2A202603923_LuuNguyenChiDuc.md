# Báo cáo cá nhân - Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603923
- **Họ và tên:** Lưu Nguyễn Chí Đức
- **SEED** (in ở ô đầu notebook): 7860

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) - notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) - do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc_1.jpg`](hand_calc_1.jpg), [`hand_calc_2.jpg`](hand_calc_2.jpg) - ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

LLM chạy qua OpenRouter: model nhỏ `qwen/qwen3-30b-a3b-instruct-2507`, model lớn `qwen/qwen3-235b-a22b-2507`.
Hàng LLM chỉ đo trên SUB 40 tin.

| Cách | F1 test | F1 SUB 40 | p95 ms/tin | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg (tự code) | 0.875 | 0.919 | 0.05 | 0 |
| 2. BoW + MLP | 0.928 | 0.923 | 0.19 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.884 | 0.974 | 38.1 | 0 |
| 3b. Encoder (MiniLM) + LogReg | 0.938 | 0.974 | 17.0 | 0 |
| 3c. Encoder (MiniLM) + MLP head | 0.933 | 0.950 | 27.0 | 0 |
| 4a. LLM nhỏ zero-shot | - | 0.950 | 1107 | 0.008 |
| 4b. LLM lớn zero-shot | - | 0.974 | 1245 | 0.011 |
| 4c. LLM nhỏ few-shot | - | 0.950 | 1189 | 0.019 |

## 3. Dự đoán và thực tế

**`dl_relu`:** Mình đoán bỏ ReLU thì mạng "tương đương MLP".
Thực tế 2 lớp Linear liên tiếp gộp lại thành một phép tuyến tính, tức là tương đương logistic regression.
F1 vẫn 0.928, bằng MLP có ReLU, nghĩa là phần hơn Phần 1 (0.875) đến từ cách train (Adam, mini-batch), không phải từ tính phi tuyến.

**`ml_lr`:** Mình đoán `lr=50` làm đường loss "bẻ cong" và hỏng.
Thực tế loss vọt lên ~2.6 ở vài epoch đầu rồi hội tụ về 0.005, thấp nhất trong 3 mức lr (lr=0.5: 0.083, lr=0.001: 0.627).
Loss của logistic regression là hàm lồi nên bước lớn chỉ dao động lúc đầu, không phân kỳ; nhưng train loss thấp như vậy là gần học thuộc tập train.

**`hard_eval`:** Mình đoán BoW sẽ làm tệ ở nhóm "Tưởng spam" và "Nói về spam".
Thực tế BoW đạt 100% ở cả hai nhóm, nhưng chỉ 10% / 0% ở "Spam giọng tự nhiên" và 17% / 0% ở "Né từ khoá".
Lý do: BoW đoán ham cho gần như mọi tin lạ (accuracy tổng 50-52%), nên nó "đúng" ở các nhóm có nhãn ham và sai gần hết các nhóm spam không có từ khoá quen thuộc.
Ngược lại, LLM nhỏ không cần nhãn mà vẫn đạt F1 0.935 trên case khó, trong khi 3b chỉ 0.717.

## 4. Phân tích 50 case khó

BoW tụt nhiều nhất: BoW + MLP từ F1 0.923 xuống 0.324, BoW + LogReg từ 0.919 xuống 0.333.
Embedding giảm còn ~0.72, LLM gần như giữ nguyên (0.950 → 0.935).
BoW + MLP đúng 0% ở nhóm "Spam giọng tự nhiên" và "Né từ khoá".

- **#29** "C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd...": BoW + MLP đoán ham. Các token đã bị đổi chữ thành số nên không có trong vocab 2000 từ, vector gần như rỗng, mô hình không thấy tín hiệu spam nào.
- **#14** "Hi, I'm working from home and making about £300 a day...": BoW + MLP đoán ham. Tin viết giọng cá nhân, toàn từ phổ biến kéo về ham ("i", "me", "you"), không có từ khoá spam quen thuộc.

BoW đạt 100% ở nhóm "Tưởng spam" và "Nói về spam" chỉ vì nó đoán ham cho gần như mọi tin (accuracy tổng 50-52%).
LLM với prompt phòng thủ (4d) lại sai ở nhóm "Tưởng spam" (#1, #2, #9, #10): prompt làm model nghi ngờ quá mức các tin có "WON", "FREE", "Call".

## 5. Thử thách cá nhân

Đề #1: Đổi ngưỡng quyết định 0.5 → 0.2 và → 0.8 cho mô hình Phần 1, báo precision/recall/F1 từng ngưỡng, chọn ngưỡng cho ngân hàng sợ chặn nhầm tin OTP.

*(TODO: điền cách làm, bảng kết quả, giải thích)*

## 6. Phương án cho sản phẩm thật

Đạt p95 < 50 ms: BoW + LogReg (0.05 ms), BoW + MLP (0.19 ms), 3a (38.1 ms), 3b (17.0 ms), 3c (27.0 ms).
LLM (~1.1-1.2 s) không đạt.
Chọn **3b Encoder + LogReg**: đạt SLA, F1 test cao nhất (0.938), F1 SUB 0.974, và trên 50 case khó tốt hơn BoW nhiều (0.717 so với ~0.33).
Đánh đổi: phải chạy encoder MiniLM (cần nhiều worker hoặc GPU để đạt ~116 tin/s), và vẫn yếu với spam giọng tự nhiên.
Nếu dùng LLM nhỏ cho toàn bộ: 10 triệu tin/ngày × $0.0075/1k ≈ $75/ngày ≈ $2,250/tháng, và vẫn không đạt độ trễ.
Thiết kế lai: 3b xử lý mọi tin, chỉ tin "lưỡng lự" (p ∈ [0.3, 0.7]) được gửi sang LLM xử lý bất đồng bộ, giữ được độ trễ cho phần lớn tin và tận dụng độ bền của LLM (F1 0.935 trên case khó) với chi phí nhỏ.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | AI trả lời/viết gì | Mình đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| 0-3 | Review câu trả lời phần 0-3 | Chỉ ra câu thiếu số liệu/cơ chế, gợi ý ý chính bằng tiếng Việt | Tự viết lại bằng lời của mình, kiểm tra số trong output |
| 4 | Viết TODO 4.1-4.4, chuyển sang OpenRouter với model rẻ | Code `call_openai`, prompt, `parse_label`, `llm_classify_batch`; cấu hình Qwen3 | Đọc lại code, chạy ô kiểm tra (✅), chạy thật trên SUB |
| 4-5 | Gợi ý trả lời Câu hỏi 4, 5, 6 và báo cáo | Bản nháp dựa trên output của mình | Đối chiếu số liệu, viết lại |

Điều học được: điểm cao trên tập test cùng phân phối chưa chứng minh mô hình hiểu bài toán; BoW đạt F1 0.92 nhưng chỉ còn 0.33 trên case khó.
So sánh mô hình cần nhìn cả chất lượng, độ trễ và chi phí cùng lúc (đường Pareto), và chênh lệch nhỏ trên tập 40 tin không đủ để xếp hạng.
