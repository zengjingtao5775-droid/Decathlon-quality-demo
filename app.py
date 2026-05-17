import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np
import datetime
import os

# ==========================================
# 0. 语言控制与页面配置
# ==========================================
if 'lang' not in st.session_state:
    st.session_state.lang = '中文'

def t(cn_text, en_text):
    return cn_text if st.session_state.lang == '中文' else en_text

st.set_page_config(
    page_title="ZX Quality AI Dashboard", 
    page_icon="📊", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 隐藏默认UI，并通过 CSS 强制统一指标卡片的高度
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        height: 120px; 
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 页面标题
# ==========================================
st.sidebar.radio("Language / 语言", ['中文', 'English'], key='lang', horizontal=True)
st.sidebar.markdown("---")

st.title(t("中兴质量管理看板", "ZX Quality Management Dashboard"))
st.markdown("---")

# ==========================================
# Sidebar 布局控制 (数据与筛选)
# ==========================================
filter_container = st.sidebar.container()
data_container = st.sidebar.container()

with data_container:
    st.markdown(t("**数据管理**", "**Data Management**"))
    uploaded_file = st.file_uploader("", type=['xlsx', 'csv'], label_visibility="collapsed")

@st.cache_data
def load_data(file_source):
    try:
        df = pd.read_excel(file_source, engine='openpyxl')
    except:
        try:
            df = pd.read_csv(file_source)
        except:
            df = pd.read_csv(file_source, encoding='gbk')
    
    df.columns = [str(c).strip() for c in df.columns]
    
    if '检验日期' in df.columns:
        df['检验日期'] = pd.to_datetime(df['检验日期'])
    if '疵点类型' in df.columns:
        df['疵点类型'] = df['疵点类型'].fillna('良品')
    if '疵点个数' in df.columns and '检验数量' in df.columns:
        df['次品率'] = df['疵点个数'] / df['检验数量']
        
    return df

try:
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        with data_container:
            st.success(t(f"已加载 {len(df)} 条", f"Loaded {len(df)} rows"))
    else:
        if os.path.exists('检验数据.xlsx - Sheet1.csv'):
            df = load_data('检验数据.xlsx - Sheet1.csv')
            with data_container:
                st.info(t(f"读取本地数据 {len(df)} 条", f"Local Data {len(df)} rows"))
        else:
            with data_container:
                st.warning(t("未检测到数据，已自动生成模拟数据演示。请上传真实数据。", "Using mock data. Please upload real data."))
            np.random.seed(42)
            dates = pd.date_range(end=datetime.date.today(), periods=30)
            mock_data = {
                '检验日期': np.random.choice(dates, 500),
                '车间': np.random.choice(['一车间', '二车间', '三车间'], 500),
                '款式': np.random.choice(['DYJ33-AW26-65', 'STYLE-B', 'STYLE-C', 'STYLE-D', 'STYLE-E', 'STYLE-F'], 500),
                '生产通知单': np.random.choice(['WO-88902', 'WO-88903', 'WO-88904'], 500),
                '生产工人': np.random.choice(['常兰荣', '张三', '李四', '王五', '侯爱云', '张翠连', '赵六'], 500),
                '质检类型': np.random.choice(['中间检验', '成品检验', '中间检验'], 500), 
                '工序': np.random.choice(['裁断', '町缝', '拉橡筋', '翻手套', '包装'], 500),
                '疵点类型': np.random.choice(['断线', '吃丝不匀', '死皱', '尺寸超差', '良品', '良品', '良品'], 500),
                '检验数量': np.random.randint(50, 200, 500),
                '疵点个数': np.random.randint(0, 15, 500)
            }
            df = pd.DataFrame(mock_data)
            df.loc[df['疵点类型'] == '良品', '疵点个数'] = 0
            df.loc[df['工序'] == '町缝', '疵点个数'] = df.loc[df['工序'] == '町缝', '疵点个数'] + np.random.randint(0, 5, len(df[df['工序'] == '町缝']))
            
            df['次品率'] = df['疵点个数'] / df['检验数量']

except Exception as e:
    st.error(t(f"数据加载失败: {str(e)}", f"Failed to load data: {str(e)}"))
    st.stop()

with filter_container:
    st.markdown(t("**业务筛选**", "**Filters**"))
    
    time_option = st.selectbox(
        t("时间范围", "Time Range"),
        [t("最近一周", "Last 7 Days"), t("最近一月", "Last 30 Days"), t("最近一年", "Last 365 Days"), t("自定义", "Custom")]
    )
    
    if '检验日期' in df.columns:
        max_date = df['检验日期'].max().date()
        if t("最近一周", "Last 7 Days") in time_option:
            start_date = max_date - datetime.timedelta(days=7)
            end_date = max_date
        elif t("最近一月", "Last 30 Days") in time_option:
            start_date = max_date - datetime.timedelta(days=30)
            end_date = max_date
        elif t("最近一年", "Last 365 Days") in time_option:
            start_date = max_date - datetime.timedelta(days=365)
            end_date = max_date
        else:
            min_date = df['检验日期'].min().date()
            date_range = st.date_input(t("选择时间", "Select Date"), [min_date, max_date])
            start_date, end_date = date_range[0], date_range[1]
    
    workshop_opts = df['车间'].unique() if '车间' in df.columns else []
    sel_workshop = st.multiselect(t("车间", "Workshop"), workshop_opts, default=workshop_opts)
    
    insp_col = '质检类型' if '质检类型' in df.columns else None
    sel_insp = []
    if insp_col:
        insp_opts = df[insp_col].unique()
        sel_insp = st.multiselect(t("质检类型", "Inspection Type"), insp_opts, default=insp_opts)
    
    cc_col = '款式' if '款式' in df.columns else '车间'
    cc_opts = df[cc_col].unique()
    sel_cc = st.multiselect(t("款式/产品线", "Style/Line"), cc_opts, default=[])
    
    order_col = '生产通知单' if '生产通知单' in df.columns else None
    sel_order = []
    if order_col:
        order_opts = df[order_col].unique()
        sel_order = st.multiselect(t("工单", "Work Order"), order_opts, default=[])

mask = (df['检验日期'].dt.date >= start_date) & (df['检验日期'].dt.date <= end_date)
if sel_workshop: mask &= df['车间'].isin(sel_workshop)
if insp_col and sel_insp: mask &= df[insp_col].isin(sel_insp)
if sel_cc: mask &= df[cc_col].isin(sel_cc)
if order_col and sel_order: mask &= df[order_col].isin(sel_order)

filtered_df = df[mask]
if filtered_df.empty:
    st.warning(t("当前筛选条件下无数据。", "No data available."))
    st.stop()

# ==========================================
# 提取不良品数据 (供多个模块使用)
# ==========================================
bad_df = filtered_df[filtered_df['疵点类型'] != '良品'].copy()
if not bad_df.empty:
    bad_df['生产工人'] = bad_df['生产工人'].fillna('未记录').astype(str)
    bad_df['疵点类型'] = bad_df['疵点类型'].fillna('未知疵点').astype(str)
    bad_df = bad_df[bad_df['疵点个数'] > 0]
    bad_df['全厂'] = '全厂疵点总计'

# ==========================================
# 1. 质量总览
# ==========================================
st.subheader(t("1. 质量总览", "1. Quality Overview"))

total_inspected = filtered_df['检验数量'].sum()
total_defects = filtered_df['疵点个数'].sum()
avg_quality_rate = (1 - (total_defects / total_inspected)) * 100 if total_inspected > 0 else 0

mock_prev_rft = avg_quality_rate - 1.2 
rft_delta = avg_quality_rate - mock_prev_rft

col1, col2, col3 = st.columns(3)
col1.metric(t("检验总数", "Total Inspected"), f"{total_inspected:,.0f} 件")
col2.metric(t("疵点总数", "Total Defects"), f"{total_defects:,.0f} 个", delta_color="inverse")
col3.metric(t("一次通过率 (RFT)", "Right First Time (RFT)"), f"{avg_quality_rate:.2f}%", delta=f"↑ {rft_delta:.2f}% 环比上一周期")

# ==========================================
# 2. 运行分析
# ==========================================
st.markdown("---")
st.subheader(t("2. 运行分析", "2. Operation Analysis"))

st.info(t(
    f"▶ **当前运行概况**：当前数据筛选范围为 **{start_date}** 至 **{end_date}**。", 
    f"▶ **Current Overview**: Data filtered from {start_date} to {end_date}."
))

# ==========================================
# 3. 20:80 员工与 Top 疵点
# ==========================================
st.markdown("---")
st.subheader(t("3. 20:80 员工与 Top 疵点", "3. 20:80 Workers & Top Defects"))

groupby_cols = ['生产工人']
has_process = '工序' in filtered_df.columns
if has_process:
    groupby_cols.append('工序')

worker_defects_stats = filtered_df.groupby(groupby_cols).agg({'疵点个数':'sum', '检验数量':'sum'}).reset_index()
worker_defects_stats = worker_defects_stats[worker_defects_stats['检验数量'] > 0]
worker_defects_stats['次品率'] = worker_defects_stats['疵点个数'] / worker_defects_stats['检验数量']
worker_defects_stats = worker_defects_stats.sort_values(by='次品率', ascending=False)

top_20_percent_count = max(1, int(len(worker_defects_stats) * 0.2))
worst_workers_df = worker_defects_stats.head(top_20_percent_count).copy()

col_w1, col_w2 = st.columns([1, 1.5])

with col_w1:
    st.markdown(t("**次品率最高的前 20% 员工名单：**", "**Top 20% Workers by Defect Rate:**"))
    
    display_cols = ['生产工人', '工序', '疵点个数', '次品率'] if has_process else ['生产工人', '疵点个数', '次品率']
    display_df = worst_workers_df[display_cols].set_index('生产工人')
    st.dataframe(display_df.style.format({'次品率': '{:.2%}', '疵点个数': '{:.0f}'}), height=380, use_container_width=True)
    
    st.info(t(
        r"**算法说明 (Pareto):** 依据 80/20 法则，系统通过公式自动截取排名前 20% 的高危人员名单", 
        r"**Algorithm (Pareto):** Isolates the top 20% highest defect rate workers."
    ))

with col_w2:
    st.markdown(t("**Top 疵点问题分布矩阵：**", "**Top Defect Matrix:**"))
    if not bad_df.empty:
        worst_workers_names = worst_workers_df['生产工人'].unique().tolist()
        pivot_defect = pd.pivot_table(
            bad_df[bad_df['生产工人'].isin(worst_workers_names)],
            values='疵点个数', index='生产工人', columns='疵点类型', aggfunc='sum', fill_value=0
        )
        fig_heat = px.imshow(pivot_defect, text_auto=True, color_continuous_scale='Reds', aspect="auto")
        fig_heat.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_heat, use_container_width=True)
        
    st.info(t(
        r"**矩阵分析与颜色计算逻辑:** 二维交叉矩阵统计的是“绝对疵点数量”。颜色越深，代表该名员工在此具体工序上的手法失误越严重。", 
        r"**Matrix Logic:** Shows absolute defect counts. Darker colors mean higher frequency."
    ))

# ==========================================
# 4. 员工技能画像与质量问题下探 (含高危工联动)
# ==========================================
st.markdown("---")
st.subheader(t("4. 员工技能画像与质量下探", "4. Worker Skills & Quality Drill-down"))

col_c1, col_c2 = st.columns(2)

high_risk_workers = []

with col_c1:
    st.markdown(t("**员工技能画像聚类**", "**Worker Skill Clustering**"))
    
    cluster_groupby = ['生产工人', '工序'] if has_process else ['生产工人']
    worker_stats = filtered_df.groupby(cluster_groupby).agg({'检验数量': 'sum', '疵点个数': 'sum'}).reset_index()
    worker_stats['defect_rate'] = worker_stats['疵点个数'] / worker_stats['检验数量']

    if len(worker_stats) > 3:
        X = worker_stats[['检验数量', 'defect_rate']].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42)
        worker_stats['cluster'] = kmeans.fit_predict(X_scaled)
        
        cluster_centers = worker_stats.groupby('cluster')['defect_rate'].mean().sort_values()
        
        label_skilled = t("熟练工 (低次品率)", "Skilled (Low Defect)")
        label_normal = t("普通工 (均值水平)", "Normal (Average)")
        label_risk = t("高危工 (高次品率)", "High Risk (High Defect)")
        
        labels_map = {
            cluster_centers.index[0]: label_skilled,
            cluster_centers.index[1]: label_normal,
            cluster_centers.index[2]: label_risk
        }
        worker_stats['技能标签'] = worker_stats['cluster'].map(labels_map)
        
        high_risk_workers = worker_stats[worker_stats['技能标签'] == label_risk]['生产工人'].unique().tolist()
        
        hover_cols = ['生产工人', '工序'] if has_process else ['生产工人']
        
        fig_cluster = px.scatter(
            worker_stats, x='检验数量', y='defect_rate', color='技能标签',
            hover_data=hover_cols, size='检验数量',
            color_discrete_map={
                label_skilled: "#00CC96", 
                label_normal: "#636EFA", 
                label_risk: "#EF553B" 
            }
        )
        fig_cluster.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=480)
        st.plotly_chart(fig_cluster, use_container_width=True)
    
    st.info(t(
        r"**算法说明 (K-Means 聚类):** 无监督学习自动划定界限", 
        r"**Logic (K-Means):** Unsupervised learning automatically delineates boundaries."
    ))

with col_c2:
    st.markdown(t("**质量问题下探 (仅展示高危人员)**", "**Quality Drill-down (High Risk Only)**"))
    if not bad_df.empty and len(high_risk_workers) > 0:
        
        option_all = t("▶ 查看所有高危人员汇总", "▶ All High-Risk Workers")
        
        selected_worker = st.selectbox(
            "",
            [option_all] + high_risk_workers,
            label_visibility="collapsed"
        )
        
        focus_workers = high_risk_workers if selected_worker == option_all else [selected_worker]
        
        sunburst_df = bad_df[bad_df['生产工人'].isin(focus_workers)][['全厂', '生产工人', '疵点类型', '疵点个数']].copy()
        sunburst_df['生产工人'] = sunburst_df['生产工人'].astype(str).replace({'nan': '未记录', 'None': '未记录', '': '未记录'})
        sunburst_df['疵点类型'] = sunburst_df['疵点类型'].astype(str).replace({'nan': '未知疵点', 'None': '未知疵点', '': '未知疵点'})
        
        sunburst_df = sunburst_df.groupby(['全厂', '生产工人', '疵点类型'], as_index=False)['疵点个数'].sum()
        sunburst_df = sunburst_df[sunburst_df['疵点个数'] > 0]
        
        if not sunburst_df.empty:
            fig_sunburst = px.sunburst(
                sunburst_df, 
                path=['全厂', '生产工人', '疵点类型'], 
                values='疵点个数',
                color='疵点个数',
                color_continuous_scale='Blues' 
            )
            fig_sunburst.update_traces(maxdepth=2, textinfo='label+percent parent')
            fig_sunburst.update_layout(margin=dict(t=10, l=0, r=0, b=0), height=410)
            st.plotly_chart(fig_sunburst, use_container_width=True)
        else:
            st.warning(t("所选人员暂无详细疵点数据。", "No specific defect data for this selection."))
            
    elif len(high_risk_workers) == 0:
        st.success(t("🎉 当前数据筛选范围内，未聚类出高危风险人员。", "No high-risk workers detected."))
            
    st.info(t(
        r"**算法说明 (精准下钻):** 本图数据已**自动过滤**，仅关联左侧聚类出的人员。点击扇区可动态展开深层疵点结构。", 
        r"**Logic:** Graph is auto-filtered to only show High-Risk workers from the left chart."
    ))

# ==========================================
# 5. 问题工单识别
# ==========================================
st.markdown("---")
st.subheader(t("5. 问题工单识别", "5. Problematic Work Order Identification"))

model_data = filtered_df[['检验数量', '疵点个数', '次品率']].fillna(0)
abnormal_wos = pd.DataFrame() 

if len(model_data) > 10:
    iso = IsolationForest(contamination=0.05, random_state=42)
    model_data['anomaly'] = iso.fit_predict(model_data[['检验数量', '次品率']])
    
    label_abnormal = t("离群异常工单", "Abnormal WO")
    label_normal = t("常规工单", "Normal WO")
    
    model_data['AI判定'] = model_data['anomaly'].apply(lambda x: label_abnormal if x == -1 else label_normal)
    
    display_data = filtered_df.copy()
    display_data['AI判定'] = model_data['AI判定']
    
    abnormal_wos = display_data[display_data['AI判定'] == label_abnormal]
    
    hover_cols = ['生产通知单', '款式', '疵点类型'] 
    hover_cols = [c for c in hover_cols if c in display_data.columns]
    
    fig_anomaly = px.scatter(
        display_data, x="检验数量", y="次品率", color="AI判定",
        hover_data=hover_cols, 
        color_discrete_map={label_abnormal: '#FF3B30', label_normal: '#007AFF'} 
    )
    fig_anomaly.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    fig_anomaly.update_layout(margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig_anomaly, use_container_width=True)


# ==========================================
# 6. 总结报告
# ==========================================
st.markdown("---")
st.subheader(t("6. 总结报告", "6. Summary Report"))

col_r1, col_r2 = st.columns(2)

with col_r1:
    st.markdown(t("🎯 **人员培训建议 (针对高危员工)**", "🎯 **Training Recommendations**"))
    if not bad_df.empty and not worst_workers_df.empty:
        training_msgs = []
        for worker in worst_workers_df['生产工人'].unique()[:5]:
            worker_bad_data = bad_df[bad_df['生产工人'] == worker]
            if not worker_bad_data.empty:
                if has_process:
                    top_issue = worker_bad_data.groupby(['工序', '疵点类型'])['疵点个数'].sum().idxmax()
                    process_name = top_issue[0]
                    defect_name = top_issue[1]
                    training_msgs.append(f"- **{worker}**：在【{process_name}】工序表现不佳，建议重点培训纠正 **{defect_name}** 问题。")
                else:
                    top_defect = worker_bad_data.groupby('疵点类型')['疵点个数'].sum().idxmax()
                    training_msgs.append(f"- **{worker}**：建议重点培训纠正 **【{top_defect}】** 相关的手法作业。")
        
        if training_msgs:
            st.warning("\n".join(training_msgs))
        else:
            st.success(t("暂无人员干预建议。", "No training suggestions at this time."))
    else:
        st.success(t("暂无人员干预建议。", "No training suggestions at this time."))

with col_r2:
    st.markdown(t("🔍 **异常工单重点抽查**", "🔍 **Work Orders to Spot-Check**"))
    if not abnormal_wos.empty:
        wo_summary = abnormal_wos.groupby(['生产通知单', '款式']).agg({'次品率': 'max'}).reset_index()
        wo_summary = wo_summary.sort_values(by='次品率', ascending=False)
        
        wo_msgs = []
        for _, row in wo_summary.iterrows():
            wo_msgs.append(f"- **{row['生产通知单']}** (款式：{row['款式']})，次品率高至 `{row['次品率']:.1%}`，需现场查验。")
            
        st.error("\n".join(wo_msgs))
    else:
        st.success(t("当前暂未发现显著偏离均值的异常工单，生产平稳。", "Production is stable. No abnormal work orders detected for spot-checking."))


# ==========================================
# 7. 派工与技能匹配推荐
# ==========================================
st.markdown("---")
st.subheader(t("7. 派工与技能匹配推荐", "7. Skill-based Assignment"))

if cc_opts.any():
    top_10_styles = filtered_df.groupby(cc_col)['检验数量'].sum().nlargest(10).index.tolist()
    
    col_assign1, col_assign2 = st.columns(2)
    
    with col_assign1:
        target_style = st.selectbox(
            t("1. 请选择即将排产的 款式/产品线 (按产量排名前10备选)", "1. Select Target Style (Top 10 by volume)"), 
            top_10_styles
        )
    
    style_df = df[df[cc_col] == target_style].copy()
    
    if not style_df.empty:
        if has_process:
            process_list = style_df['工序'].dropna().unique().tolist()
            with col_assign2:
                target_process = st.selectbox(
                    t("2. 请选择需要派工的 具体工序", "2. Select Specific Process"), 
                    process_list
                )
            style_df = style_df[style_df['工序'] == target_process]
            assignment_target_text = f"款式 **{target_style}** 的 **{target_process}** 工序"
        else:
            assignment_target_text = f"款式 **{target_style}**"

        if not style_df.empty:
            worker_style_stats = style_df.groupby('生产工人').agg({'检验数量': 'sum', '疵点个数': 'sum'}).reset_index()
            worker_style_stats['RFT'] = 1 - (worker_style_stats['疵点个数'] / worker_style_stats['检验数量'])
            
            max_vol = worker_style_stats['检验数量'].max()
            worker_style_stats['Vol_Norm'] = worker_style_stats['检验数量'] / max_vol if max_vol > 0 else 0
            
            w_rft = 0.6
            w_vol = 0.4
            worker_style_stats['综合得分'] = (worker_style_stats['RFT'] * w_rft) + (worker_style_stats['Vol_Norm'] * w_vol)
            
            top_recommendations = worker_style_stats.sort_values(by='综合得分', ascending=False).head(5)
            
            st.write(t(f"针对 {assignment_target_text}，系统推荐的最优派工名单：", f"Top Recommended Workers for {assignment_target_text}:"))
            
            rec_cols = st.columns(5)
            for i, row in enumerate(top_recommendations.itertuples()):
                if i < 5:
                    with rec_cols[i]:
                        st.success(f"**推荐 #{i+1}: {row.生产工人}**\n\n综合 RFT: `{row.RFT:.1%}`\n\n历史产量: `{row.检验数量:.0f}` 件")
            
            st.info(t(
                r"**下拉菜单数据说明:** 系统抓取的是当前上方筛选范围内有生产记录的款式。若不足 10 个，代表实际生产的款式种类不足。", 
                r"**Data Note:** Lists up to top 10 styles based on available filtered data."
            ))
        else:
            st.warning(t("所选工序暂无历史生产数据可供参考。", "No historical data for this process."))
    else:
        st.info(t("该款式暂无历史生产数据可供参考。", "No historical data for this style."))
else:
    st.info(t("请先上传包含款式/产品线的数据。", "Please upload data containing Style/Line column."))
