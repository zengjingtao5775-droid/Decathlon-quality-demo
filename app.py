import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# ==========================================
# 1. 页面配置与标题
# ==========================================
st.set_page_config(page_title="ZTE Quality AI Hub", page_icon="🛡️", layout="wide")

st.title("🛡️ 中兴质量Dashboard")
st.markdown("### 基于 Python & AI 的全链路质量数据分析系统")

# ==========================================
# 2. Sidebar 布局控制 (关键修改：调整顺序)
# ==========================================
# 定义两个容器，控制视觉顺序
# filter_container 在上，data_container 在下
filter_container = st.sidebar.container()
st.sidebar.markdown("---")
data_container = st.sidebar.container()

# ==========================================
# 3. 数据加载与预处理 (在 data_container 中渲染)
# ==========================================
with data_container:
    st.header("📂 数据管理")
    uploaded_file = st.file_uploader("📤 上传最新质量数据 (Excel/CSV)", type=['xlsx', 'csv'])

@st.cache_data
def load_data(file_source):
    try:
        df = pd.read_excel(file_source, engine='openpyxl')
    except:
        try:
            df = pd.read_csv(file_source)
        except:
            df = pd.read_csv(file_source, encoding='gbk')
    
    # 清洗逻辑
    df.columns = [str(c).strip() for c in df.columns]
    
    if '检验日期' in df.columns:
        df['检验日期'] = pd.to_datetime(df['检验日期'])
    
    if '疵点类型' in df.columns:
        df['疵点类型'] = df['疵点类型'].fillna('良品')
        
    if '疵点个数' in df.columns and '检验数量' in df.columns:
        df['次品率'] = df['疵点个数'] / df['检验数量']
        
    return df

# 数据加载逻辑
try:
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        with data_container:
            st.success(f"✅ 已加载上传数据：{len(df)} 条")
    else:
        default_path = '检验数据.xlsx - Sheet1.csv'
        df = load_data(default_path)
        with data_container:
            st.info(f"ℹ️ 使用本地演示数据：{len(df)} 条")
except Exception as e:
    st.error(f"❌ 数据加载失败。请上传文件或检查本地文件路径。错误信息: {e}")
    st.stop()

# ==========================================
# 4. 交互式筛选 (在 filter_container 中渲染) - [改进点1]
# ==========================================
with filter_container:
    st.header("🔍 交互式筛选")
    
    # 动态获取选项
    workshop_options = df['车间'].unique() if '车间' in df.columns else []
    selected_workshop = st.multiselect(
        "选择车间", workshop_options, default=workshop_options
    )

    if '检验日期' in df.columns:
        min_date = df['检验日期'].min()
        max_date = df['检验日期'].max()
        date_range = st.date_input(
            "选择时间段", [min_date, max_date]
        )
    else:
        st.warning("数据缺少'检验日期'列")
        st.stop()

# 应用筛选
mask = (df['车间'].isin(selected_workshop)) & \
       (df['检验日期'].dt.date >= date_range[0]) & \
       (df['检验日期'].dt.date <= date_range[1])
filtered_df = df[mask]

if filtered_df.empty:
    st.warning("当前筛选条件下无数据，请调整筛选器。")
    st.stop()

# ==========================================
# 5. 核心指标看板
# ==========================================
total_inspected = filtered_df['检验数量'].sum()
total_defects = filtered_df['疵点个数'].sum()
avg_quality_rate = (1 - (total_defects / total_inspected)) * 100 if total_inspected > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("总检验数量", f"{total_inspected:,.0f} 件")
col2.metric("总疵点数", f"{total_defects:,.0f} 个", delta_color="inverse")
col3.metric("整体良品率", f"{avg_quality_rate:.2f}%", delta="目标 > 98%")
col4.metric("AI 监控模型", "在线运行", delta="4个模型")

# ==========================================
# 6. [模块 A] 工人画像 (保持 AI 逻辑说明)
# ==========================================
st.markdown("---")
st.subheader("1. 工人技能画像聚类 (K-Means Clustering)")

worker_stats = filtered_df.groupby('生产工人').agg({
    '检验数量': 'sum',
    '疵点个数': 'sum'
}).reset_index()
worker_stats['defect_rate'] = worker_stats['疵点个数'] / worker_stats['检验数量']

if len(worker_stats) > 3:
    X = worker_stats[['检验数量', 'defect_rate']].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    worker_stats['cluster'] = kmeans.fit_predict(X_scaled)
    
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
    
    st.info("""
    ℹ️ **AI 评估逻辑说明**：
    此模型使用 **K-Means 聚类算法**，基于工人的 **“历史产出总量” (X轴)** 和 **“平均次品率” (Y轴)** 两个维度进行自动分层：
    - **🌟 熟练工 (绿色)**：产出高且次品率低
    - **🔧 普通工 (蓝色)**：产出和质量处于平均水平
    - **⚠️ 待培训 (红色)**：次品率显著偏高，或产出极低，建议安排针对性工艺培训
    """)
else:
    st.info("数据不足以进行聚类分析")

# ==========================================
# 7. [模块 B] 基础图表
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

# ==========================================
# 8. [模块 C] Sankey 归因分析 - [改进点2]
# ==========================================
st.subheader("🔗 质量归因流向 (Sankey Diagram)")

# 增加专用筛选器
col_sankey_filter, _ = st.columns([1, 2])
with col_sankey_filter:
    # 获取当前数据下的所有款式
    available_styles = ['全部'] + list(filtered_df['款式'].astype(str).unique())
    selected_style_sankey = st.selectbox("🎯 筛选特定款式 (减少线束干扰)", available_styles)

# 数据准备
sankey_raw = filtered_df[filtered_df['疵点类型']!='良品']
if selected_style_sankey != '全部':
    sankey_raw = sankey_raw[sankey_raw['款式'].astype(str) == selected_style_sankey]

# 取 Top 50 防止渲染卡顿
sankey_df = sankey_raw.head(50)

if not sankey_df.empty:
    fig_sankey = px.parallel_categories(
        sankey_df, 
        dimensions=['车间', '生产工人', '疵点类型'],
        color="疵点个数",
        color_continuous_scale=px.colors.sequential.Inferno
    )
    st.plotly_chart(fig_sankey, use_container_width=True)
    
    # [改进点2] 统一格式的逻辑说明
    st.info("""
    ℹ️ **图表逻辑与应用说明**：
    - **逻辑**：展示质量问题的流动路径，从 **车间** ➡️ **责任工人** ➡️ **疵点类型**。线条粗细代表问题数量。
    - **应用**：
        1. **定位源头**：一眼看出哪个车间是问题的“重灾区”。
        2. **锁定责任人**：追踪特定疵点（如线迹不良）主要集中在哪些工人身上。
        3. **精准培训**：若某位工人只产生特定一种疵点，说明其该项工艺手法存在误区，需定向指导。
    """)
else:
    st.warning("当前筛选条件下无次品数据，无法生成流向图。")

# ==========================================
# 9. [模块 D] 异常侦测 - [改进点3]
# ==========================================
st.markdown("---")
st.subheader("2. AI 异常工单侦测 (Anomaly Detection)")

model_data = filtered_df[['检验数量', '疵点个数', '次品率']].fillna(0)

if len(model_data) > 10:
    iso = IsolationForest(contamination=0.05, random_state=42)
    model_data['anomaly'] = iso.fit_predict(model_data[['检验数量', '次品率']])
    model_data['AI判定'] = model_data['anomaly'].apply(lambda x: '🔴 异常' if x == -1 else '🔵 正常')
    
    display_data = filtered_df.copy()
    display_data['AI判定'] = model_data['AI判定']
    
    hover_cols = ['生产通知单', '生产工人', '款式', '疵点类型'] 
    hover_cols = [c for c in hover_cols if c in display_data.columns]
    
    fig_anomaly = px.scatter(
        display_data, 
        x="检验数量", y="次品率", 
        color="AI判定",
        hover_data=hover_cols,
        color_discrete_map={'🔴 异常': 'red', '🔵 正常': 'blue'},
        title="工单异常分布雷达"
    )
    
    # [改进点3] 增大点的尺寸，使其更清晰
    fig_anomaly.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    fig_anomaly.layout.yaxis.tickformat = ',.1%'
    
    st.plotly_chart(fig_anomaly, use_container_width=True)
    st.info("💡 提示：鼠标悬停在红点上，可直接查看到具体的 **“生产通知单”** 号，方便线下立即调取该批次工单进行复核。")
else:
    st.warning("数据量太少，AI 无法启动异常检测")

# ==========================================
# 10. [模块 E] 多维下钻
# ==========================================
st.subheader("3. 质量问题多维下钻")
sunburst_df = filtered_df[filtered_df['疵点类型'] != '良品'].copy()
sunburst_df['车间'] = sunburst_df['车间'].fillna("未知车间")
sunburst_df['生产工人'] = sunburst_df['生产工人'].fillna("未知工人")
sunburst_df['疵点类型'] = sunburst_df['疵点类型'].fillna("未知类型")

if not sunburst_df.empty:
    fig_sun = px.sunburst(
        sunburst_df, 
        path=['车间', '生产工人', '疵点类型'], 
        values='疵点个数',
        title="点击扇区可展开细节",
        height=600
    )
    st.plotly_chart(fig_sun, use_container_width=True)

# ==========================================
# 11. [模块 F] 班组风险模拟器
# ==========================================
st.markdown("---")
st.subheader("4. 排产风险模拟 (Team Risk Simulator)")
st.caption("模拟 **多名员工** 组合生产某一款式时的潜在质量风险。")

col_sim1, col_sim2 = st.columns([1, 2])

with col_sim1:
    st.info("👈 请在下方组建班组")
    sim_workers_list = df['生产工人'].dropna().unique()
    sim_style_list = df['款式'].unique()
    
    if len(sim_workers_list) > 0:
        sim_workers = st.multiselect("拟派工班组 (可多选)", sim_workers_list, default=[sim_workers_list[0]])
        sim_style = st.selectbox("生产款式", sim_style_list)
        sim_qty = st.slider("计划单人生产数量", 100, 5000, 2000)
    else:
        st.warning("无工人数据")
        st.stop()

with col_sim2:
    if not sim_workers:
        st.warning("请至少选择一名工人。")
    else:
        try:
            # 模型训练
            train_df = df[['生产工人', '款式', '订单数量', '次品率']].dropna()
            
            le_worker = LabelEncoder()
            le_style = LabelEncoder()
            
            train_df['worker_str'] = train_df['生产工人'].astype(str)
            train_df['style_str'] = train_df['款式'].astype(str)
            
            le_worker.fit(train_df['worker_str'])
            le_style.fit(train_df['style_str'])
            
            train_df['worker_code'] = le_worker.transform(train_df['worker_str'])
            train_df['style_code'] = le_style.transform(train_df['style_str'])
            
            rf = RandomForestRegressor(n_estimators=20, random_state=42)
            rf.fit(train_df[['worker_code', 'style_code', '订单数量']], train_df['次品率'])
            
            # 批量预测
            risk_results = []
            input_style_code = le_style.transform([str(sim_style)])[0]
            
            for worker in sim_workers:
                try:
                    w_code = le_worker.transform([str(worker)])[0]
                    pred = rf.predict([[w_code, input_style_code, sim_qty]])[0]
                    risk_results.append({'工人': worker, '风险': pred})
                except:
                    risk_results.append({'工人': worker, '风险': train_df['次品率'].mean()})
            
            risk_df = pd.DataFrame(risk_results)
            avg_risk = risk_df['风险'].mean()
            max_risk = risk_df['风险'].max()
            risky_worker = risk_df.loc[risk_df['风险'].idxmax(), '工人']
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = avg_risk * 100,
                title = {'text': f"班组平均风险 ({len(sim_workers)}人)"},
                gauge = {
                    'axis': {'range': [None, 10]},
                    'bar': {'color': "royalblue"},
                    'steps': [
                        {'range': [0, 2], 'color': "lightgreen"},
                        {'range': [2, 5], 'color': "yellow"},
                        {'range': [5, 10], 'color': "red"}],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 9.9}
                }
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            if max_risk > 0.05:
                st.error(f"⚠️ **班组短板预警**：\n虽然平均风险为 {avg_risk:.1%}, 但工人 **{risky_worker}** 在该款式的预测风险高达 **{max_risk:.1%}**。建议将其替换或安排技术指导。")
            else:
                st.success(f"✅ **班组配置合理**：\n所有成员预测风险均在可控范围内 (最高 {max_risk:.1%})。")
                
            with st.expander("查看每位成员的详细预测值"):
                st.dataframe(risk_df.style.format({"风险": "{:.2%}"}))
                
        except Exception as e:
            st.warning(f"预测计算中遇到未知数据，无法精确模拟。Details: {e}")
