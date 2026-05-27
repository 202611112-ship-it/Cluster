import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---------------- 한글 폰트 ----------------
font_path = "NanumGothic.otf"
fontprop = fm.FontProperties(fname=font_path)

plt.rc('font', family=fontprop.get_name())
plt.rcParams['axes.unicode_minus'] = False

# ---------------- 페이지 설정 ----------------
st.set_page_config(
    page_title="폐암 시스템",
    page_icon="🌈",
    layout="wide"
)

# ---------------- CSS 꾸미기 ----------------
st.markdown("""
<style>

/* 전체 배경 */
.stApp {
    background: linear-gradient(
        45deg,
        #ff66cc,
        #66ccff,
        #ffff66,
        #99ff99,
        #ff9999
    );
    background-size: 400% 400%;
}

/* 제목 */
h1 {
    color: hotpink !important;
    text-align: center;
    font-size: 55px !important;
    font-family: Comic Sans MS;
    text-shadow: 3px 3px yellow;
}

/* 일반 글씨 */
p, label, div {
    font-family: Comic Sans MS;
    font-size: 20px !important;
}

/* 버튼 */
.stButton>button {
    background-color: yellow;
    color: purple;
    border-radius: 20px;
    border: 5px dashed hotpink;
    font-size: 28px;
    font-weight: bold;
    height: 80px;
    width: 100%;
}

/* 입력창 */
.stNumberInput input {
    background-color: #fff8dc;
    color: blue;
    font-size: 24px;
    border: 3px solid hotpink;
}

/* 성공 메시지 */
.stSuccess {
    background-color: pink;
    color: darkblue;
    border: 4px dotted purple;
    font-size: 30px;
}

/* 그래프 영역 */
.css-1d391kg {
    background-color: rgba(255,255,255,0.5);
    border-radius: 30px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- 이미지 ----------------
st.image(
    "https://i.namu.wiki/i/8l3CqU5UoM5JtQk3L0.png",
    width=250
)

# ---------------- 데이터 불러오기 ----------------
df = pd.read_csv("lung.csv")

# ---------------- 모델 불러오기 ----------------
model = joblib.load("lung_model.pkl")
scaler = joblib.load("lung_scaler.pkl")

# ---------------- 제목 ----------------
st.title("🌈 폐암 위험 군집 시스템 🌈")

st.markdown("""
<h2 style='text-align:center; color:blue;'>
🐟 건강 🐟
</h2>
""", unsafe_allow_html=True)

st.write("생활 습관 정보를 입력하면 군집을 예측해줍니다...")

# ---------------- 입력창 ----------------
col1, col2, col3 = st.columns(3)

with col1:
    smoke = st.number_input("🚬 흡연량", min_value=0.0, step=0.1)

with col2:
    environment = st.number_input("🌫️ 지역환경지수", min_value=0.0, step=0.1)

with col3:
    soju = st.number_input("🍺 음주량", min_value=0.0, step=0.1)

# ---------------- 버튼 ----------------
if st.button("✨ 군집 예측하기 ✨"):

    # 새 데이터 생성
    new_patient = pd.DataFrame(
        [[smoke, environment, soju]],
        columns=['흡연량', '지역환경지수', '음주량']
    )

    # 스케일링
    new_patient_scaled = scaler.transform(new_patient)

    # 예측
    pred_cluster = model.predict(new_patient_scaled)

    # 결과 출력
    st.balloons()

    st.success(
        f"🌟 이 환자는 {pred_cluster[0]}번 군집에 속합니다!! 🌟"
    )

    # ---------------- 그래프 ----------------
    fig, ax = plt.subplots(figsize=(10, 7))

    # 배경색
    fig.patch.set_facecolor('pink')
    ax.set_facecolor('#fffacd')

    # 기존 데이터
    ax.scatter(
        df['지역환경지수'],
        df['음주량'],
        c=df['cluster'],
        s=120,
        alpha=0.7
    )

    # 새 환자
    ax.scatter(
        environment,
        soju,
        c='red',
        s=500,
        marker='X',
        label=' 새 환자 '
    )

    # 제목
    ax.set_title(
        ' 폐암 위험 군집 시각화 ',
        fontsize=25,
        color='purple'
    )

    # 축 이름
    ax.set_xlabel(' 지역환경지수', fontsize=18)
    ax.set_ylabel(' 음주량', fontsize=18)

    # 격자
    ax.grid(True, linestyle='--', alpha=0.5)

    # 범례
    ax.legend(fontsize=15)

    # 출력
    st.pyplot(fig)

    # 마지막 감성멘트
    st.markdown("""
    <h1 style='color:blue; text-align:center;'>
    🐚 폐 건강을 지킵시다 🐚
    </h1>
    """, unsafe_allow_html=True)
