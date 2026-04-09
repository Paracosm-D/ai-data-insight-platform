import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI

# -------------------------- 1. 页面基础配置与Deepseek客户端初始化 --------------------------
st.set_page_config(page_title="AI 智能数据洞察平台", layout="wide")
st.title("📊 AI 数据分析 + 自定义数据面板平台")


# 安全读取API Key（适配线上部署）
@st.cache_resource
def init_deepseek_client():
    return OpenAI(
        api_key=st.secrets.get("DEEPSEEK_API_KEY", "sk-48fa3970badc468bb9940a6f7d453262"),
        base_url="https://api.deepseek.com/v1"
    )


client = init_deepseek_client()

# -------------------------- 2. 侧边栏导航 --------------------------
st.sidebar.title("场景导航")
analysis_mode = st.sidebar.radio(
    "请选择场景:",
    ("商业运营分析", "游戏用户数据分析", "软件/网站留存分析")
)
st.write(f"当前场景：**{analysis_mode}**")

# -------------------------- 3. 数据上传 + 基础分析 --------------------------
uploaded_file = st.file_uploader("上传 CSV 数据文件", type=['csv'])
df = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # 基础数据展示
    with st.expander("📋 原始数据概览", expanded=False):
        st.dataframe(df.head())
        st.write(f"数据规模：{df.shape[0]} 行 {df.shape[1]} 列")
        st.dataframe(df.describe())

    # ====================== 【新增核心功能】自定义数据面板搭建 ======================
    st.divider()
    st.subheader("🎨 自定义数据面板（Dashboard）搭建")
    columns = df.columns.tolist()

    # 1. 选择面板布局
    layout_option = st.radio("选择面板布局", ["单栏(100%)", "双栏(50%/50%)", "三栏(33%/33%/33%)"], horizontal=True)
    col_map = {
        "单栏(100%)": st.columns(1),
        "双栏(50%/50%)": st.columns(2),
        "三栏(33%/33%/33%)": st.columns(3)
    }
    panel_cols = col_map[layout_option]

    # 2. 面板配置：指标卡 + 自定义图表
    for i, col in enumerate(panel_cols):
        with col:
            st.markdown(f"### 面板区域 {i + 1}")
            # 选择展示内容
            card_type = st.selectbox(f"选择展示内容", ["核心指标卡", "自定义图表", "数据表格"], key=f"card_{i}")

            if card_type == "核心指标卡":
                # 自动生成统计指标
                target_col = st.selectbox("选择指标列", columns, key=f"metric_{i}")
                if pd.api.types.is_numeric_dtype(df[target_col]):
                    st.metric(label=f"📈 {target_col} 平均值", value=f"{df[target_col].mean():.2f}")
                    st.metric(label=f"📉 {target_col} 最大值", value=f"{df[target_col].max():.2f}")
                else:
                    st.warning("请选择数值型列生成指标卡")

            elif card_type == "自定义图表":
                # 自由选择X/Y轴+图表类型
                x = st.selectbox("X轴", columns, key=f"x_{i}")
                y = st.selectbox("Y轴", columns, key=f"y_{i}")
                chart = st.radio("图表类型", ["柱状图", "折线图", "散点图"], key=f"chart_{i}", horizontal=True)

                # 生成图表
                try:
                    if chart == "柱状图":
                        fig = px.bar(df, x=x, y=y)
                    elif chart == "折线图":
                        fig = px.line(df, x=x, y=y)
                    else:
                        fig = px.scatter(df, x=x, y=y)
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.error("字段选择不支持生成图表")

            elif card_type == "数据表格":
                st.dataframe(df.head(10), use_container_width=True)

    # -------------------------- 原有功能：可视化探索 --------------------------
    st.divider()
    st.subheader("📈 全局可视化探索")
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X轴", columns)
    with col2:
        y_axis = st.selectbox("Y轴", columns)
    chart_type = st.radio("图表类型", ["折线图", "柱状图", "散点图"], horizontal=True)

    if chart_type == "折线图":
        st.plotly_chart(px.line(df, x=x_axis, y=y_axis), use_container_width=True)
    elif chart_type == "柱状图":
        st.plotly_chart(px.bar(df, x=x_axis, y=y_axis), use_container_width=True)
    else:
        st.plotly_chart(px.scatter(df, x=x_axis, y=y_axis), use_container_width=True)

    # -------------------------- 原有功能：AI商业洞察 --------------------------
    st.divider()
    st.subheader("🤖 AI智能分析报告")
    if st.button("生成分析报告"):
        with st.spinner("AI分析中..."):
            try:
                data_summary = df[[x_axis, y_axis]].describe().to_string()
                prompt = f"""
                场景：{analysis_mode}，X轴：{x_axis}，Y轴：{y_axis}
                数据摘要：{data_summary}
                请输出专业的业务洞察和3条落地建议。
                """
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "system", "content": "你是专业数据分析师"}, {"role": "user", "content": prompt}],
                    temperature=0.7
                )
                st.success("分析完成！")
                st.write(response.choices[0].message.content.strip())
            except Exception as e:
                st.error(f"失败：{str(e)}")

else:
    st.info("👆 请上传CSV文件开始使用~ 支持所有表格数据，面试演示超方便！")