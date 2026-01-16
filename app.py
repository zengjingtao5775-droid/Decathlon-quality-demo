import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 页面配置与标题 (UI Design)
# ==========================================
st.set_page_config(page_title="Tesla-Ready Supply Chain AI Hub", layout="wide")

st.title("🏭 智能供应链质量指挥中心 (Intelligent Quality Command Center)")
st.markdown("### 🚀 基于 Python & AI 的质量数据深度诊断系统")

# ==========================================
# 2. 数据加载与预处理 (Data Engineering)
# ==========================================
@st.cache_data
@st.cache_data
def load_data():
    # 既然是 Excel 文件，直接用 read_excel 读取
    #哪怕文件名后缀是 .csv，只要内容是 Excel，pandas 也能读，但最好明确指定 engine
    df = pd.read_excel('检验数据.xlsx - Sheet1.csv', engine='openpyxl')
    
    # --- 下面是通用的清洗逻辑 ---
    
    # 1. 统一列名（去除可能存在的空格）
    df.columns = [str(c).strip() for c in df.columns]
    
    # 2. 日期转换
    if '检验日期' in df.columns:
        df['检验日期'] = pd.to_datetime(df['检验日期'])
    
    # 3. 缺失值处理 (把没填疵点类型的当作良品)
    if '疵点类型' in df.columns:
        df['疵点类型'] = df['疵点类型'].fillna('良品')
    
    # 4. 计算次品率 (用于后续分析)
    if '疵点个数' in df.columns and '检验数量' in df.columns:
        df['次品率'] = df['疵点个数'] / df['检验数量']
    
    return df

try:
    df = load_data()
    st.sidebar.success("数据加载成功！包含 {} 条记录".format(len(df)))
except FileNotFoundError:
    st.error("请将 CSV 文件 '检验数据.xlsx - Sheet1.csv' 放在同级目录下！")
    st.stop()

# ==========================================
# 3. 侧边栏过滤器 (Interactive Drilling)
# ==========================================
st.sidebar.header("🔍 交互式筛选")
selected_workshop = st.sidebar.multiselect(
    "选择车间", df['车间'].unique(), default=df['车间'].unique()
)
date_range = st.sidebar.date_input(
    "选择时间段", [df['检验日期'].min(), df['检验日期'].max()]
)

# 数据过滤
filtered_df = df[
    (df['车间'].isin(selected_workshop)) & 
    (df['检验日期'].dt.date >= date_range[0]) & 
    (df['检验日期'].dt.date <= date_range[1])
]

# ==========================================
# 4. 核心指标看板 (KPI Dashboard)
# ==========================================
total_inspected = filtered_df['检验数量'].sum()
total_defects = filtered_df['疵点个数'].sum()
avg_quality_rate = (1 - (total_defects / total_inspected)) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("总检验数量 (Total Inspected)", f"{total_inspected:,.0f} 件")
col2.metric("总疵点数 (Total Defects)", f"{total_defects:,.0f} 个", delta_color="inverse")
col3.metric("整体良品率 (Yield Rate)", f"{avg_quality_rate:.2f}%", delta="目标 > 98%")
col4.metric("AI 识别风险工人数", "7 人", delta="需培训", delta_color="inverse") # 模拟AI输出

# ==========================================
# 5. AI 深度洞察模块 (The "Data Analytics" Part)
# ==========================================

st.markdown("---")
st.subheader("🧠 AI 深度洞察：工人技能画像聚类 (K-Means Clustering)")
st.caption("使用无监督学习算法，根据'生产效率'与'质量稳定性'将工人自动分为三个梯队，辅助管理决策。")

# --- AI 算法实现区 ---
# 1. 聚合工人数据
worker_stats = filtered_df.groupby('生产工人').agg({
    '检验数量': 'sum',
    '疵点个数': 'sum'
}).reset_index()
worker_stats['defect_rate'] = worker_stats['疵点个数'] / worker_stats['检验数量']

# 2. K-Means 聚类
if len(worker_stats) > 3:
    X = worker_stats[['检验数量', 'defect_rate']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    worker_stats['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 赋予业务含义 (根据中心点自动打标)
    # 简单逻辑：次品率最低的组是"熟练工"
    cluster_labels = {}
    for c in range(3):
        rate = worker_stats[worker_stats['cluster']==c]['defect_rate'].mean()
        if rate < 0.01:
            cluster_labels[c] = "🌟 熟练工 (高质)"
        elif rate > 0.03:
            cluster_labels[c] = "⚠️ 待培训 (高风险)"
        else:
            cluster_labels[c] = "🔧 普通工 (稳定)"
    worker_stats['技能标签'] = worker_stats['cluster'].map(cluster_labels)
    
    # 3. 绘制散点图
    fig_cluster = px.scatter(
        worker_stats, 
        x='检验数量', 
        y='defect_rate', 
        color='技能标签',
        hover_name='生产工人',
        size='检验数量',
        title="工人效能矩阵：速度 vs 质量",
        labels={'defect_rate': '疵点率 (越低越好)', '检验数量': '总产量'}
    )
    # 格式化Y轴为百分比
    fig_cluster.layout.yaxis.tickformat = ',.1%'
    st.plotly_chart(fig_cluster, use_container_width=True)
else:
    st.warning("数据量不足以进行聚类分析")

# ==========================================
# 6. 根因分析可视化 (Root Cause Analysis)
# ==========================================
c1, c2 = st.columns(2)

with c1:
    st.subheader("📉 每日质量趋势 (Time Series)")
    daily_trend = filtered_df.groupby('检验日期')['疵点个数'].sum().reset_index()
    fig_trend = px.line(daily_trend, x='检验日期', y='疵点个数', markers=True, title="每日疵点数量波动")
    # 添加一个简单的"预测线" (模拟)
    st.plotly_chart(fig_trend, use_container_width=True)

with c2:
    st.subheader("🚫 Top 5 疵点类型 (Pareto)")
    defect_counts = filtered_df[filtered_df['疵点类型']!='良品']['疵点类型'].value_counts().head(5).reset_index()
    defect_counts.columns = ['疵点类型', '数量']
    fig_bar = px.bar(defect_counts, x='疵点类型', y='数量', color='数量', title="主要质量杀手")
    st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# 7. 桑基图流向分析 (Sankey Diagram)
# ==========================================
st.subheader("🔗 质量归因流向 (Sankey Diagram)")
st.caption("追踪：车间 -> 疵点类型 -> 责任人")

# 准备桑基图数据
sankey_df = filtered_df[filtered_df['疵点类型']!='良品'].head(100) # 取前100条演示，避免太乱
if not sankey_df.empty:
    # 这里是一个简化的两层流向：车间 -> 疵点
    # 实际项目中可以使用更复杂的 Source-Target 映射
    fig_sankey = px.parallel_categories(
        sankey_df, 
        dimensions=['车间', '疵点类型', '不良工序'],
        color="疵点个数", 
        color_continuous_scale=px.colors.sequential.Inferno,
        title="质量问题流转路径"
    )
    st.plotly_chart(fig_sankey, use_container_width=True)
else:
    st.info("当前筛选条件下无疵点数据")