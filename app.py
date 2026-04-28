#! importing Data Science Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

st.set_page_config(
    page_title="Traffic Accident Insights",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main > div {
        padding-top: 1.2rem;
    }
    .hero {
        background: linear-gradient(135deg, #0f172a, #1d4ed8);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.18);
    }
    .hero h1 {
        margin: 0;
        font-size: 1.8rem;
    }
    .hero p {
        margin: 0.3rem 0 0;
        opacity: 0.92;
    }
    .block-title {
        margin-top: 0.4rem;
        margin-bottom: 0.2rem;
    }
    div[data-testid="stMetric"] {
        background-color: transparent;
        border: 1px solid rgba(148, 163, 184, 0.35);
        padding: 0.6rem 0.8rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
    }
    div.stButton > button {
        border-radius: 10px;
        border: 1px solid #1d4ed8;
        background: #1d4ed8;
        color: white;
        font-weight: 600;
    }
    div.stButton > button:hover {
        background: #1e40af;
        border-color: #1e40af;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#! importing Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

#! import the data
df = pd.read_csv("traffic_accidents.csv")

st.markdown(
    """
    <div class="hero">
        <h1>Traffic Accidents Dashboard</h1>
        <p>Explore crash patterns, injury trends, and severity predictions in one place.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("Use this app to analyze accidents and estimate crash severity.")
st.sidebar.info("Dataset source file: `traffic_accidents.csv`")

st.header("Data Cleaning & Preprocessing")

st.write("=== Data set info ===")
st.write(df.dtypes)
st.write()
st.write("=== Missing values ===")
st.write(df.isnull().sum())

# Drop nulls
df = df.dropna()
st.write(f"Shape after dropping nulls: {df.shape}")

# Parse CrashDate and extract features
df['crash_date'] = pd.to_datetime(df['crash_date'])
df['date'] = df['crash_date'].dt.date
df['time'] = df['crash_date'].dt.time
df['year'] = df['crash_date'].dt.year
df = df.drop(columns = 'crash_date')

# Binary encode target: 0 = no injury, 1 = injury / severe
df['crash_type'] = df['crash_type'].apply (
  lambda x: 0 if x == 'NO INJURY / DRIVE AWAY' else 1
)
st.write("Target distribution: ")
st.write(df['crash_type'].value_counts())
st.write()
st.dataframe(df.head())

overview_cols = st.columns(4)
with overview_cols[0]:
    st.metric("Total Records", f"{len(df):,}")
with overview_cols[1]:
    st.metric("Severe Crash Rate", f"{df['crash_type'].mean() * 100:.1f}%")
with overview_cols[2]:
    st.metric("Total Injuries", f"{int(df['injuries_total'].sum()):,}")
with overview_cols[3]:
    st.metric("Total Fatalities", f"{int(df['injuries_fatal'].sum()):,}")

st.divider()

#! EDA (Exploratory Data Analysis)
st.markdown("## Exploratory Data Analysis")
#* Temporal Patterns

#? Bars (by hour, by day, by month)
fig, axes = plt.subplots(1, 3, figsize = (18, 5))

# By hour
hour_counts = df['crash_hour'].value_counts().sort_index()
axes[0].bar(hour_counts.index, hour_counts.values, color = 'steelblue')
axes[0].set_title('Crashes by Hour of Day')
axes[0].set_xlabel('Hour (0, 23)')
axes[0].set_ylabel('Count')

# By Day
day_labels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
day_counts = df['crash_day_of_week'].value_counts().sort_index()
axes[1].bar(day_labels, day_counts.values, color = 'coral')
axes[1].set_title('Crashes by Day of Week')
axes[1].set_xlabel('Day')

# By Month
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_counts = df['crash_month'].value_counts().sort_index()
axes[2].bar(month_labels, month_counts.values, color = 'mediumseagreen')
axes[2].set_title('Crashes by Month')
axes[2].set_xlabel('Month')
plt.xticks(rotation = 45)

plt.tight_layout()
st.pyplot(fig)

#? Heatmap (Hour vs Day of Week)
pivot = df.groupby(['crash_day_of_week', 'crash_hour']).size().unstack(fill_value = 0)
pivot.index = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

fig = plt.figure(figsize=(16, 5))
sns.heatmap(pivot, cmap='YlOrRd', linewidths=0.3)
plt.title('Crash frequency - Hour of Day vs Day of Week')
plt.xlabel('Hour of Day')
plt.ylabel('Day of Week')
st.pyplot(fig)

#* Environmental conditions
st.subheader("Environmental Conditions")
fig = plt.figure(figsize=(12, 5))
sns.countplot(x=df['weather_condition'], data=df, order=df['weather_condition'].value_counts().index[:10])
plt.xticks(rotation=45, ha='right')
plt.title('Accidents by Weather Condition (Top 10)')
plt.tight_layout()
st.pyplot(fig)

fig = plt.figure(figsize=(12, 5))
sns.countplot(x='weather_condition', hue='crash_type', data=df, order=df['weather_condition'].value_counts().index[:8])
plt.xticks(rotation=45, ha='right')
plt.title('Accidents Severity by Weather Condition')
plt.legend(title='Crash Type', labels=['No Injury', 'Injury / Severe'])
plt.tight_layout()
st.pyplot(fig)

fig = plt.figure(figsize=(10, 5))
sns.countplot(x='lighting_condition', data=df, order=df['lighting_condition'].value_counts().index)
plt.xticks(rotation=45, ha='right')
plt.title('Accidents by Lighting Condition')
plt.tight_layout()
st.pyplot(fig)

#* Road & Crash Type
st.subheader("Road & Crash Type")
fig = plt.figure(figsize=(12, 5))
sns.countplot(x='first_crash_type', data=df,
              order=df['first_crash_type'].value_counts().index[:10])
plt.xticks(rotation=45, ha='right')
plt.title('Accidents by First Crash Type (Top 10)')
plt.tight_layout()
st.pyplot(fig)

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
sns.countplot(ax=axes[0], x='trafficway_type', data=df, order=df['trafficway_type'].value_counts().index[:8])
axes[0].set_title('Accidents by Road Type')
axes[0].tick_params(axis='x', rotation=45)

sns.countplot(ax=axes[1], x='trafficway_type', hue='crash_type',data=df, order=df['trafficway_type'].value_counts().index[:6])
axes[1].set_title('Accidents Severity by Road Type')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(title='Crash Type', labels=['No Injury', 'Injury / Severe'])

plt.tight_layout()
st.pyplot(fig)

#? Injuries & Severity
st.subheader("Injuries & Severity")
injury_cols = ['injuries_total', 'injuries_fatal', 'injuries_incapacitating',
               'injuries_non_incapacitating', 'injuries_reported_not_evident', 'injuries_no_indication']
st.write("=== Injury totals across all crashes ===")
st.write(df[injury_cols].sum().astype(int))
st.write()
st.write("=== Average per crash ===")
st.write(df[injury_cols].mean().round(4))

severity_counts = df['most_severe_injury'].value_counts()

fig = plt.figure(figsize=(8, 6))
plt.pie(severity_counts.values, labels=severity_counts.index,
        autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Most Severe Injury')
plt.tight_layout()
st.pyplot(fig)

#? Contriburting Causes
st.subheader("Contributing Causes")
cause_counts = df['prim_contributory_cause'].value_counts().head(12)
fig = plt.figure(figsize=(12, 6))
cause_counts[::-1].plot(kind='barh', color='tomato')
plt.title('Top 12 Primary Contributing Causes')
plt.xlabel('Number of Crashes')
plt.tight_layout()
st.pyplot(fig)

#! Correlation matrix
st.header("Correlation Analysis")
num_cols = ['num_units', 'injuries_total', 'injuries_fatal',
            'injuries_incapacitating', 'injuries_non_incapacitating',
            'injuries_reported_not_evident', 'injuries_no_indication',
            'crash_hour', 'crash_day_of_week', 'crash_month', 'crash_type']

corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

fig = plt.figure(figsize=(12, 8))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1, linewidths=0.5)
plt.title('Correlation Matrix')
plt.tight_layout()
st.pyplot(fig)

#! Crash Severity Prediction
st.header("Crash Severity Prediction Models")
st.subheader("Feature Encoding & Train/Test Split")
features = [
    'weather_condition',
    'lighting_condition',
    'crash_hour',
    'trafficway_type',
    'first_crash_type'
]

# Use a separate dict of encoders so we can decode later
encoders = {}
df_model = df[features + ['crash_type']].copy()

categorical_features = ['weather_condition', 'lighting_condition', 'trafficway_type', 'first_crash_type']

for col in categorical_features:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    encoders[col] = le  # save each encoder separately

# Display encoding maps
with st.expander("📋 View Encoding Maps"):
    for col, le in encoders.items():
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        st.markdown(f"**{col}:**")
        for k, v in sorted(mapping.items()):
            st.write(f"  {v} = {k}")

X = df_model[features]
y = df_model['crash_type']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

col1, col2 = st.columns(2)
with col1:
    st.metric("Training Samples", f"{len(X_train):,}")
with col2:
    st.metric("Test Samples", f"{len(X_test):,}")


#? DecisionTree Specifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

st.subheader("Decision Tree Results")
dt_acc = accuracy_score(y_test, y_pred_dt)
st.metric("Accuracy", f"{dt_acc:.4f}")
st.markdown("**Classification Report:**")
dt_report = classification_report(y_test, y_pred_dt, target_names=['No Injury', 'Injury/Severe'])
st.code(dt_report)

#? Confusion matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=['No Injury', 'Injury/Severe'])

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap='Blues', ax=ax)
ax.set_title('Decision Tree — Confusion Matrix')
fig.tight_layout()
st.pyplot(fig)

#? RandomForest Specifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

st.subheader("Random Forest Results")
rf_acc = accuracy_score(y_test, y_pred_rf)
st.metric("Accuracy", f"{rf_acc:.4f}")
st.markdown("**Classification Report:**")
rf_report = classification_report(y_test, y_pred_rf, target_names=['No Injury', 'Injury/Severe'])
st.code(rf_report)

cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['No Injury', 'Injury/Severe'])

fig, ax = plt.subplots(figsize=(6, 5))
disp_rf.plot(cmap='Greens', ax=ax)
ax.set_title('Random Forest — Confusion Matrix')
fig.tight_layout()
st.pyplot(fig)

#! Model Comparison
st.subheader("Model Comparison")

models  = ['Decision Tree', 'Random Forest']
scores  = [dt_acc, rf_acc]
colors  = ['steelblue', 'mediumseagreen']

fig = plt.figure(figsize=(7, 4))
bars = plt.bar(models, scores, color=colors, width=0.4)
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.02,
             f'{score:.4f}', ha='center', va='top', fontsize=13, color='white', fontweight='bold')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.tight_layout()
st.pyplot(fig)

winner = 'Random Forest' if rf_acc > dt_acc else 'Decision Tree'
st.success(f'🏆 Best model: **{winner}** ({max(scores):.4f})')

#? Feature importance
st.subheader("Feature Importance")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax, model, name, color in [
    (axes[0], dt_model,  'Decision Tree',  'steelblue'),
    (axes[1], rf_model,  'Random Forest',  'mediumseagreen')
]:
    imp_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)

    ax.barh(imp_df['Feature'], imp_df['Importance'], color=color)
    ax.set_title(f'{name} — Feature Importance')
    ax.set_xlabel('Importance Score')

plt.tight_layout()
st.pyplot(fig)

#! Live Prediction (Using Random Forest)
st.header("Live Prediction")
st.write("Use the controls below to predict if a crash will be severe using the Random Forest model.")

st.markdown("### Encoding Reference:")
encoding_cols = st.columns(len(encoders))
for col_idx, (col_name, le) in enumerate(encoders.items()):
    with encoding_cols[col_idx]:
        st.markdown(f"**{col_name}:**")
        for i, cls in enumerate(le.classes_):
            st.write(f"{i} = {cls}")

st.markdown("### Make a Prediction:")
pred_cols = st.columns(5)

with pred_cols[0]:
    weather = st.selectbox('Weather condition', range(len(encoders['weather_condition'].classes_)),
                           format_func=lambda x: f"{x}: {encoders['weather_condition'].classes_[x]}")

with pred_cols[1]:
    lighting = st.selectbox('Lighting condition', range(len(encoders['lighting_condition'].classes_)),
                            format_func=lambda x: f"{x}: {encoders['lighting_condition'].classes_[x]}")

with pred_cols[2]:
    hour = st.slider('Crash hour (0–23):', 0, 23, 12)

with pred_cols[3]:
    road = st.selectbox('Trafficway type', range(len(encoders['trafficway_type'].classes_)),
                        format_func=lambda x: f"{x}: {encoders['trafficway_type'].classes_[x]}")

with pred_cols[4]:
    crash_t = st.selectbox('First crash type', range(len(encoders['first_crash_type'].classes_)),
                           format_func=lambda x: f"{x}: {encoders['first_crash_type'].classes_[x]}")

if st.button("🔮 Make Prediction", key="predict_btn"):
    sample = [[weather, lighting, hour, road, crash_t]]
    prediction = rf_model.predict(sample)
    probability = rf_model.predict_proba(sample)[0]

    col1, col2 = st.columns(2)
    with col1:
        if prediction[0] == 1:
            st.error("⚠️ Prediction: SEVERE ACCIDENT")
        else:
            st.success("✅ Prediction: MINOR / NO INJURY")

    with col2:
        st.metric("Confidence", f"{max(probability):.1%}")


#! Key insights summary
st.header("📊 Key Insights Summary")

peak_hour  = df['crash_hour'].mode()[0]
peak_day   = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'][df['crash_day_of_week'].mode()[0]]
top_weather = df['weather_condition'].mode()[0]
top_cause   = df['prim_contributory_cause'].mode()[0]
fatal_count = (df['injuries_fatal'] > 0).sum()
severe_pct  = df['crash_type'].mean() * 100

# Display metrics in rows
st.markdown("### Overview")
metric_cols = st.columns(4)
with metric_cols[0]:
    st.metric("Total Records", f"{len(df):,}")
with metric_cols[1]:
    st.metric("Severe Crash Rate", f"{severe_pct:.1f}%")
with metric_cols[2]:
    st.metric("Total Injuries", f"{int(df['injuries_total'].sum()):,}")
with metric_cols[3]:
    st.metric("Total Fatalities", f"{int(df['injuries_fatal'].sum()):,}")

st.markdown("### Peak Times & Conditions")
insight_cols = st.columns(3)
with insight_cols[0]:
    st.info(f"**Peak Hour:** {peak_hour}:00 (24-hour format)")
with insight_cols[1]:
    st.info(f"**Peak Day:** {peak_day}")
with insight_cols[2]:
    st.info(f"**Most Common Weather:** {top_weather}")

st.markdown("### Top Contributing Factor")
st.warning(f"🚗 {top_cause}")

st.markdown("### Crashes with Fatality")
st.metric("Incidents", f"{fatal_count:,}")

st.markdown("### Model Performance Summary")
model_perf_cols = st.columns(2)
with model_perf_cols[0]:
    st.metric("Decision Tree Accuracy", f"{dt_acc:.4f}")
with model_perf_cols[1]:
    st.metric("Random Forest Accuracy", f"{rf_acc:.4f}")
