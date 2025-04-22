import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Hàm xử lý dữ liệu: encode tất cả cột object thành số
def preprocess_data(df):
    df_processed = df.copy()
    for col in df_processed.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    return df_processed

# Tiêu đề ứng dụng
st.title('Phân loại quantity với SuperStore Orders')

# Tải file CSV
uploaded_file = st.file_uploader("Chọn file SuperStoreOrders.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dữ liệu mẫu")
    st.dataframe(df.head())

    # Chọn thuật toán
    task = st.selectbox("Chọn thuật toán:", ["Logistic Regression", "Decision Tree Classifier"])

    # Tiền xử lý dữ liệu
    df_processed = preprocess_data(df)

    # Đảm bảo cột 'quantity' tồn tại
    if 'quantity' not in df_processed.columns:
        st.error("Cột 'quantity' không tồn tại trong dữ liệu. Vui lòng kiểm tra file CSV.")
    else:
        # Chọn các thuộc tính dự đoán
        features = st.multiselect(
            "Chọn các thuộc tính dự đoán:",
            [col for col in df_processed.columns if col != 'quantity'],
            default=[col for col in df_processed.columns if col != 'quantity'][:3]  # Mặc định chọn 3 cột đầu
        )

        if features:
            X = df_processed[features]
            y = df_processed['quantity']

            # Chuẩn hóa dữ liệu
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Chia dữ liệu thành tập train/test
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Xử lý theo thuật toán
            if task == "Logistic Regression":
                model = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Đánh giá mô hình
                accuracy = accuracy_score(y_test, y_pred)
                st.subheader("Kết quả Logistic Regression")
                st.write(f"Độ chính xác: {accuracy:.2f}")

                # Ma trận nhầm lẫn
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Dự đoán')
                ax.set_ylabel('Thực tế')
                ax.set_title('Ma trận nhầm lẫn - Logistic Regression')
                st.pyplot(fig)

            elif task == "Decision Tree Classifier":
                model = DecisionTreeClassifier(random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Đánh giá mô hình
                accuracy = accuracy_score(y_test, y_pred)
                st.subheader("Kết quả Decision Tree Classifier")
                st.write(f"Độ chính xác: {accuracy:.2f}")

                # Ma trận nhầm lẫn
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Dự đoán')
                ax.set_ylabel('Thực tế')
                ax.set_title('Ma trận nhầm lẫn - Decision Tree Classifier')
                st.pyplot(fig)

else:
    st.info("Vui lòng upload file CSV để bắt đầu.")