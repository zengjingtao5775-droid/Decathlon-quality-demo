import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# ==========================================
# 1. 页面配置与标题 (UI Design)
# ==========================================
st.set_page_config(page_title="Tesla-Ready Supply Chain AI Hub", layout="wide")

st.title("🏭 中兴质量看板")
st.markdown("### 基于 Python & AI 的质量数据诊断系统")

# ==========================================
# 2. 数据加载与预处理 (Data Engineering)
# ==========================================
@st.cache_data
def load_data():
    # 强力读取模式：专治各类 Excel/CSV 疑难杂症
    file_path = '检验数据.xlsx - Sheet1.csv'
    try:
        # 优先尝试 Excel 引擎读取
        df = pd.read_excel(file_path, engine='openpyxl')
    except:
        try:
            # 失败则尝试标准 CSV
            df = pd.read_csv(file_path)
        except:
            # 最后尝试 GBK 编码
            df = pd.read_csv(file_path, encoding='gbk')
    
    # --- 数据清洗 ---
    # 1. 统一列名（去除可能存在的空格）
    df.columns = [str(c).strip() for c in df.columns]
    
    # 2. 日期转换
    if '检验日期' in df.columns:
        df['检验日期'] = pd.to_datetime(df['检验日期'])
    
    # 3. 缺失值处理
    if '疵点类型' in df.columns:
        df['疵点类型'] = df['疵点类型'].fillna('良品')
        
    # 4. 计算次品率
    if '疵点个数' in df.columns and '检验数量' in df.columns:
        df['次品率'] = df['疵点个数'] / df['检验数量']
        
    return df

try:
    df = load_data()
    # 侧边栏显示状态
    st.sidebar.success(f"数据加载成功！共 {len(df)} 条记录")
except Exception as e:
    st.error(f"数据读取失败，请检查目录下是否有 '检验数据.xlsx - Sheet1.csv' 文件。错误信息: {e}")
    st.stop()

# ==========================================
# 3. 侧边栏过滤器 (Interactive Drilling)
# ==========================================
st.sidebar.header("🔍 交互式筛选")

# 获取选项列表
workshop_options = df['车间'].unique() if '车间' in df.columns else []
selected_workshop = st.sidebar.multiselect(
    "选择车间", workshop_options, default=workshop_options
)

# 日期筛选
min_date = df['检验日期'].min()
max_date = df['检验日期'].max()
date_range = st.sidebar.date_input(
    "选择时间段", [min_date, max_date]
)

# 应用过滤
mask = (df['车间'].isin(selected_workshop)) & \
       (df['检验日期'].dt.date >= date_range[0]) & \
       (df['检验日期'].dt.date <= date_range[1])
filtered_df = df[mask]

if filtered_df.empty:
    st.warning("当前筛选条件下无数据，请调整筛选器。")
    st.stop()

# ==========================================
# 4. 核心指标看板 (KPI Dashboard)
# ==========================================
total_inspected = filtered_df['检验数量'].sum()
total_defects = filtered_df['疵点个数'].sum()
avg_quality_rate = (1 - (total_defects / total_inspected)) * 100 if total_inspected > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("总检验数量 (Total Inspected)", f"{total_inspected:,.0f} 件")
col2.metric("总疵点数 (Total Defects)", f"{total_defects:,.0f} 个", delta_color="inverse")
col3.metric("整体良品率 (Yield Rate)", f"{avg_quality_rate:.2f}%", delta="目标 > 98%")
col4.metric("AI 监控模型", "运行中", delta="3个模型在线")

# ==========================================
# 5. [模块 A] AI 深度洞察：工人画像 (Clustering)
# ==========================================
st.markdown("---")
st.subheader("1. 工人技能画像聚类 (K-Means Clustering)")
st.caption("AI 自动将工人分为：熟练工(高质)、普通工、待培训(高风险)")

worker_stats = filtered_df.groupby('生产工人').agg({
    '检验数量': 'sum',
    '疵点个数': 'sum'
}).reset_index()
worker_stats['defect_rate'] = worker_stats['疵点个数'] / worker_stats['检验数量']

if len(worker_stats) > 3:
    X = worker_stats[['检验数量', 'defect_rate']]
    # 填充可能得 NaN
    X = X.fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    worker_stats['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 自动打标逻辑
    cluster_labels = {}
    for c in range(3):
        rate = worker_stats[worker_stats['cluster']==c]['defect_rate'].mean()
        if rate < 0.015:
            cluster_labels[c] = "🌟 熟练工 (高质)"
        elif rate > 0.035:
            cluster_labels[c] = "⚠️ 待培训 (高风险)"
        else:
            cluster_labels[c] = "🔧 普通工 (稳定)"
    worker_stats['技能标签'] = worker_stats['cluster'].map(cluster_labels)
    
    fig_cluster = px.scatter(
        worker_stats, 
        x='检验数量', y='defect_rate', 
        color='技能标签',
        hover_name='生产工人', size='检验数量',
        title="工人效能矩阵",
        color_discrete_map={"🌟 熟练工 (高质)":"green", "🔧 普通工 (稳定)":"blue", "⚠️ 待培训 (高风险)":"red"}
    )
    fig_cluster.layout.yaxis.tickformat = ',.1%'
    st.plotly_chart(fig_cluster, use_container_width=True)
else:
    st.info("数据不足以进行聚类分析")

# ==========================================
# 6. [模块 B] 基础图表 (Trend & Pareto & Sankey)
# ==========================================
c1, c2 = st.columns(2)
with c1:
    st.subheader("📉 每日质量趋势")
    daily_trend = filtered_df.groupby('检验日期')['疵点个数'].sum().reset_index()
    fig_trend = px.line(daily_trend, x='检验日期', y='疵点个数', markers=True)
    st.plotly_chart(fig_trend, use_container_width=True)

with c2:
    st.subheader("🚫 Top 5 疵点类型")
    defect_counts = filtered_df[filtered_df['疵点类型']!='良品']['疵点类型'].value_counts().head(5).reset_index()
    if not defect_counts.empty:
        defect_counts.columns = ['疵点类型', '数量']
        fig_bar = px.bar(defect_counts, x='疵点类型', y='数量', color='数量')
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.write("无疵点数据")

st.subheader("🔗 质量归因流向 (Sankey)")
sankey_df = filtered_df[filtered_df['疵点类型']!='良品'].head(50) # 限制数量防卡顿
if not sankey_df.empty:
    fig_sankey = px.parallel_categories(
        sankey_df, 
        dimensions=['车间', '疵点类型', '不良工序'],
        color="疵点个数",
        color_continuous_scale=px.colors.sequential.Inferno
    )
    st.plotly_chart(fig_sankey, use_container_width=True)

# ==========================================
# 7. [模块 C] 🕵️‍♂️ AI 异常侦测 (Isolation Forest)
# ==========================================
st.markdown("---")
st.subheader(" 2. AI 异常工单侦测 (Anomaly Detection)")
st.caption("利用孤立森林算法，自动标记出数据分布异常的工单（可能是漏检或极端次品率）。")

model_data = filtered_df[['检验数量', '疵点个数', '次品率']].fillna(0)

if len(model_data) > 10:
    # 训练模型
    iso = IsolationForest(contamination=0.05, random_state=42)
    # 预测 (-1为异常, 1为正常)
    model_data['anomaly'] = iso.fit_predict(model_data[['检验数量', '次品率']])
    model_data['AI判定'] = model_data['anomaly'].apply(lambda x: '🔴 异常' if x == -1 else '🔵 正常')
    
    # 绘图数据准备
    display_data = filtered_df.copy()
    display_data['AI判定'] = model_data['AI判定']
    
    fig_anomaly = px.scatter(
        display_data, 
        x="检验数量", y="次品率", 
        color="AI判定",
        hover_data=['生产工人', '款式', '疵点类型'],
        color_discrete_map={'🔴 异常': 'red', '🔵 正常': 'blue'},
        title="工单异常分布雷达"
    )
    fig_anomaly.layout.yaxis.tickformat = ',.1%'
    st.plotly_chart(fig_anomaly, use_container_width=True)
else:
    st.warning("数据量太少，AI 无法启动异常检测")

# ==========================================
# ==========================================
# 8. [模块 D] ☀️ 3. 质量问题多维下钻 (Sunburst)
# ==========================================
st.subheader(" 3. 质量问题多维下钻")

# 1. 筛选出瑕疵品
sunburst_df = filtered_df[filtered_df['疵点类型'] != '良品'].copy()

# 2. 关键修复：填充空值！防止出现"断枝"
# 如果车间或工人是空的，Plotly 会报错，必须填上默认值
sunburst_df['车间'] = sunburst_df['车间'].fillna("未知车间")
sunburst_df['生产工人'] = sunburst_df['生产工人'].fillna("未知工人")
sunburst_df['疵点类型'] = sunburst_df['疵点类型'].fillna("未知类型")

if not sunburst_df.empty:
    fig_sun = px.sunburst(
        sunburst_df, 
        path=['车间', '生产工人', '疵点类型'], 
        values='疵点个数',
        title="点击扇区可展开细节 (已自动修复空值数据)",
        height=600
    )
    st.plotly_chart(fig_sun, use_container_width=True)
else:
    st.info("当前筛选范围内没有次品数据，无法生成旭日图。")

# ==========================================
# 9. [模块 E]  生产风险模拟器 (Predictive Model)
# ==========================================
st.markdown("---")
st.subheader(" 4. 生产风险模拟器 (Risk Simulator)")
st.caption("基于随机森林算法，预测新任务的潜在次品率风险。")

col_sim1, col_sim2 = st.columns([1, 2])

with col_sim1:
    st.info("👈 请在下方设定参数")
    # 模拟输入
    sim_workers_list = df['生产工人'].dropna().unique()
    sim_style_list = df['款式'].unique()
    
    if len(sim_workers_list) > 0:
        sim_worker = st.selectbox("拟派工人", sim_workers_list)
        sim_style = st.selectbox("生产款式", sim_style_list)
        sim_qty = st.slider("计划数量", 100, 5000, 2000)
    else:
        st.warning("无工人数据")
        st.stop()

with col_sim2:
    # 实时训练简单模型
    try:
        # 准备训练数据
        train_df = df[['生产工人', '款式', '订单数量', '次品率']].dropna()
        
        # 简单编码
        le_worker = LabelEncoder()
        le_style = LabelEncoder()
        
        # 将所有已知标签转为字符串防止类型错误
        train_df['worker_str'] = train_df['生产工人'].astype(str)
        train_df['style_str'] = train_df['款式'].astype(str)
        
        # 训练编码器
        le_worker.fit(train_df['worker_str'])
        le_style.fit(train_df['style_str'])
        
        train_df['worker_code'] = le_worker.transform(train_df['worker_str'])
        train_df['style_code'] = le_style.transform(train_df['style_str'])
        
        # 训练模型
        rf = RandomForestRegressor(n_estimators=20, random_state=42)
        rf.fit(train_df[['worker_code', 'style_code', '订单数量']], train_df['次品率'])
        
        # 预测当前输入
        input_worker_code = le_worker.transform([str(sim_worker)])[0]
        input_style_code = le_style.transform([str(sim_style)])[0]
        
        pred_rate = rf.predict([[input_worker_code, input_style_code, sim_qty]])[0]
        
        # 仪表盘展示
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = pred_rate * 100,
            title = {'text': "AI 预测风险 (%)"},
            gauge = {
                'axis': {'range': [None, 10]},
                'bar': {'color': "royalblue"},
                'steps': [
                    {'range': [0, 2], 'color': "lightgreen"},
                    {'range': [2, 5], 'color': "yellow"},
                    {'range': [5, 10], 'color': "red"}],
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        if pred_rate > 0.05:
            st.error(f"⚠️ 高风险预警：该组合历史表现不佳！")
        else:
            st.success("✅ 风险可控：推荐该组合。")
            
    except Exception as e:
        st.warning(f"无法进行预测，可能因该工人/款式无历史数据。Details: {e}")
