%%writefile app.py
import streamlit as st
import pandas as pd
import time
import plotly.express as px
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import LinearRegression
from pyspark.ml.stat import Correlation

# --- تهيئة الصفحة والذاكرة ---
st.set_page_config(page_title="Cloud Data Service", layout="wide")

# تهيئة مخزن النتائج لتسجيل المحاولات للتقرير
if 'perf_history' not in st.session_state:
    st.session_state['perf_history'] = []
# تهيئة مخزن لأوقات الأساس (T1) لكل خوارزمية
if 'baseline_times' not in st.session_state:
    st.session_state['baseline_times'] = {}

st.title(" Cloud-Based Distributed Data Processing Service ")

# --- دالة مساعدة لحساب الأداء ---
def calculate_metrics(job_name, nodes, duration):
    # حفظ وقت الأساس إذا كان عدد الأجهزة 1
    if nodes == 1:
        st.session_state['baseline_times'][job_name] = duration
        speedup = 1.0
        efficiency = 1.0
    else:
        # محاولة جلب وقت الأساس
        t1 = st.session_state['baseline_times'].get(job_name)
        if t1:
            speedup = t1 / duration
            efficiency = speedup / nodes
        else:
            speedup = 0.0 # غير معروف
            efficiency = 0.0
    
    # إضافة النتيجة للسجل
    st.session_state['perf_history'].append({
        "Algorithm": job_name,
        "Nodes": nodes,
        "Time (s)": round(duration, 4),
        "Speedup": round(speedup, 2),
        "Efficiency": round(efficiency, 2)
    })
    
    return speedup, efficiency

# --- رفع البيانات ---
uploaded_file = st.file_uploader("1. Upload Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    with open("temp_data.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(" File uploaded to Cloud Storage!")

    # --- القائمة الجانبية ---
    st.sidebar.header("Processing Options")
    num_nodes = st.sidebar.select_slider(
        "Select Number of Worker Nodes",
        options=[1, 2, 4, 8]
    )
    
    # تهيئة Spark
    spark = SparkSession.builder \
        .appName("EngineeringProject") \
        .master(f"local[{num_nodes}]") \
        .getOrCreate()
    
    df = spark.read.csv("temp_data.csv", header=True, inferSchema=True)
    numeric_cols = [c for c, t in df.dtypes if t in ('int', 'double', 'float')]

    # --- اختيار الوظيفة ---
    job_type = st.selectbox(
        "2. Select Processing Job",
        ["Descriptive Statistics", "K-Means Clustering", "Linear Regression", "Correlation Matrix"]
    )

    # ---------------------------------------------------------
    # الوظيفة 1: الإحصاءات الوصفية
    # ---------------------------------------------------------
    if job_type == "Descriptive Statistics":
        if st.button("Run Analysis"):
            start_time = time.time()
            stats = df.describe().toPandas() # العملية الثقيلة
            end_time = time.time()
            
            duration = end_time - start_time
            sp, eff = calculate_metrics("Statistics", num_nodes, duration)
            
            st.subheader(" Results")
            st.table(stats)
            
            # عرض مقاييس الأداء
            st.markdown("---")
            st.subheader(" Performance Metrics")
            c1, c2, c3 = st.columns(3)
            c1.metric("Execution Time", f"{duration:.4f} s")
            c2.metric("Speedup", f"{sp:.2f}x", help="T1 / Tp")
            c3.metric("Efficiency", f"{eff:.2f}", help="Speedup / Nodes")
            if num_nodes > 1 and sp == 0:
                st.warning(" Run on 1 Node first to calculate Speedup!")

    # ---------------------------------------------------------
    # الوظيفة 2: K-Means
    # ---------------------------------------------------------
    elif job_type == "K-Means Clustering":
        selected_cols = st.multiselect("Select Columns", numeric_cols, default=numeric_cols[:2] if len(numeric_cols)>1 else None)
        k_val = st.slider("K", 2, 10, 3)
        
        if st.button("Run Clustering") and len(selected_cols) >= 2:
            start_time = time.time()
            assembler = VectorAssembler(inputCols=selected_cols, outputCol="features")
            data = assembler.transform(df.na.drop())
            kmeans = KMeans().setK(k_val).setSeed(1)
            model = kmeans.fit(data)
            predictions = model.transform(data)
            end_time = time.time()
            
            duration = end_time - start_time
            sp, eff = calculate_metrics("K-Means", num_nodes, duration)
            
            st.success("Clustering Completed.")
            
            # عرض مقاييس الأداء
            st.markdown("---")
            st.subheader("⏱️ Performance Metrics")
            c1, c2, c3 = st.columns(3)
            c1.metric("Execution Time", f"{duration:.4f} s")
            c2.metric("Speedup", f"{sp:.2f}x")
            c3.metric("Efficiency", f"{eff:.2f}")
            if num_nodes > 1 and sp == 0:
                st.warning(" Run on 1 Node first to calculate Speedup!")
                
            st.subheader("Visualization")
            plot_data = predictions.select(selected_cols + ['prediction']).limit(1000).toPandas()
            fig = px.scatter(plot_data, x=selected_cols[0], y=selected_cols[1], color=plot_data['prediction'].astype(str))
            st.plotly_chart(fig)

    # ---------------------------------------------------------
    # الوظيفة 3: Linear Regression
    # ---------------------------------------------------------
    elif job_type == "Linear Regression":
        target_col = st.selectbox("Target", numeric_cols)
        feature_cols = st.multiselect("Features", [c for c in numeric_cols if c != target_col])
        
        if st.button("Train Model") and feature_cols:
            start_time = time.time()
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            data = assembler.transform(df.na.drop())
            train, test = data.randomSplit([0.8, 0.2])
            lr = LinearRegression(labelCol=target_col)
            model = lr.fit(train)
            result = model.evaluate(test)
            end_time = time.time()
            
            duration = end_time - start_time
            sp, eff = calculate_metrics("Linear Regression", num_nodes, duration)
            
            st.success("Model Trained.")
            st.write(f"**R2 Score:** {result.r2:.4f}")
            
            # عرض مقاييس الأداء
            st.markdown("---")
            st.subheader(" Performance Metrics")
            c1, c2, c3 = st.columns(3)
            c1.metric("Execution Time", f"{duration:.4f} s")
            c2.metric("Speedup", f"{sp:.2f}x")
            c3.metric("Efficiency", f"{eff:.2f}")
            if num_nodes > 1 and sp == 0:
                st.warning(" Run on 1 Node first to calculate Speedup!")

    # ---------------------------------------------------------
    # الوظيفة 4: Correlation Matrix
    # ---------------------------------------------------------
    elif job_type == "Correlation Matrix":
        if st.button("Calculate Correlation"):
            start_time = time.time()
            assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
            data = assembler.transform(df.na.drop())
            pearson_corr = Correlation.corr(data, "features").head()
            corr_matrix = pearson_corr[0].toArray()
            end_time = time.time()
            
            duration = end_time - start_time
            sp, eff = calculate_metrics("Correlation", num_nodes, duration)
            
            st.subheader("Heatmap")
            df_corr = pd.DataFrame(corr_matrix, columns=numeric_cols, index=numeric_cols)
            fig = px.imshow(df_corr, text_auto=True)
            st.plotly_chart(fig)
            
            # عرض مقاييس الأداء
            st.markdown("---")
            st.subheader(" Performance Metrics")
            c1, c2, c3 = st.columns(3)
            c1.metric("Execution Time", f"{duration:.4f} s")
            c2.metric("Speedup", f"{sp:.2f}x")
            c3.metric("Efficiency", f"{eff:.2f}")
            if num_nodes > 1 and sp == 0:
                st.warning(" Run on 1 Node first to calculate Speedup!")

    # ---------------------------------------------------------
    # جدول التقرير النهائي (تراكمي)
    # ---------------------------------------------------------
    st.markdown("---")
    st.header(" Performance Report Table")
    if len(st.session_state['perf_history']) > 0:
        report_df = pd.DataFrame(st.session_state['perf_history'])
        st.table(report_df)
        st.info(" Tip: You can copy this table directly to your project report.")
        if st.button("Clear History"):
            st.session_state['perf_history'] = []
            st.session_state['baseline_times'] = {}
            st.experimental_rerun()
