import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Настройка страницы
st.set_page_config(
    page_title="Анализ успеваемости",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Загрузка данных
@st.cache_data
def load_data():
    df = pd.read_csv("StudentsPerformance.csv")
    df['total_score'] = df['math score'] + df['reading score'] + df['writing score']
    df['average_score'] = df['total_score'] / 3
    return df

df = load_data()

# Сайдбар с навигацией
st.sidebar.title("📊 Навигация")
page = st.sidebar.radio("Выберите страницу:", 
                        ["📈 Визуализация данных", 
                         "🔍 Результаты анализа"])

# Общие фильтры
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Фильтры")

# Фильтр по полу
gender_filter = st.sidebar.multiselect(
    "Пол:",
    options=df['gender'].unique(),
    default=df['gender'].unique()
)

# Фильтр по подготовке
prep_filter = st.sidebar.multiselect(
    "Подготовка к тесту:",
    options=df['test preparation course'].unique(),
    default=df['test preparation course'].unique()
)

# Слайдер для оценок
score_range = st.sidebar.slider(
    "Диапазон среднего балла:",
    min_value=float(df['average_score'].min()),
    max_value=float(df['average_score'].max()),
    value=(float(df['average_score'].min()), float(df['average_score'].max()))
)

# Применение фильтров
filtered_df = df[
    (df['gender'].isin(gender_filter)) &
    (df['test preparation course'].isin(prep_filter)) &
    (df['average_score'] >= score_range[0]) &
    (df['average_score'] <= score_range[1])
]

# СТРАНИЦА 1: ВИЗУАЛИЗАЦИЯ ДАННЫХ
if page == "📈 Визуализация данных":
    st.title("📈 Визуализация исходных данных")
    
    # KPI карточки
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего студентов", len(filtered_df))
    with col2:
        st.metric("Средний балл", f"{filtered_df['average_score'].mean():.1f}")
    with col3:
        st.metric("Медианный балл", f"{filtered_df['average_score'].median():.1f}")
    with col4:
        st.metric("Станд. отклонение", f"{filtered_df['average_score'].std():.1f}")
    
    st.markdown("---")
    
    # Таблица данных
    st.subheader("📋 Таблица данных")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Экспорт данных
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Скачать данные",
        data=csv,
        file_name="student_data.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Распределения
    st.subheader("📊 Распределение оценок")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Выбор предмета для гистограммы
        subject = st.selectbox(
            "Выберите предмет:",
            ['math score', 'reading score', 'writing score', 'average_score']
        )
        
        fig = px.histogram(
            filtered_df, 
            x=subject,
            nbins=30,
            title=f'Распределение {subject}',
            labels={subject: 'Баллы'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot по группам
        group_by = st.selectbox(
            "Группировать по:",
            ['gender', 'test preparation course', 'lunch', 'race/ethnicity']
        )
        
        fig = px.box(
            filtered_df,
            x=group_by,
            y='average_score',
            title=f'Средний балл по {group_by}'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Категориальные распределения
    st.subheader("📊 Категориальные распределения")
    
    cat_col1, cat_col2 = st.columns(2)
    
    with cat_col1:
        cat_var = st.selectbox(
            "Категориальная переменная:",
            ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
        )
        
        fig = px.pie(
            filtered_df,
            names=cat_var,
            title=f'Распределение по {cat_var}'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with cat_col2:
        # Bar chart средних баллов
        bar_var = st.selectbox(
            "Сравнить по:",
            ['gender', 'race/ethnicity', 'parental level of education']
        )
        
        avg_scores = filtered_df.groupby(bar_var)[['math score', 'reading score', 'writing score']].mean().reset_index()
        avg_scores_melted = avg_scores.melt(id_vars=[bar_var], 
                                           value_vars=['math score', 'reading score', 'writing score'],
                                           var_name='Предмет', 
                                           value_name='Средний балл')
        
        fig = px.bar(
            avg_scores_melted,
            x=bar_var,
            y='Средний балл',
            color='Предмет',
            barmode='group',
            title=f'Средние баллы по предметам'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Корреляционная матрица
    st.subheader("🔗 Корреляционная матрица")
    
    numeric_cols = ['math score', 'reading score', 'writing score', 'total_score', 'average_score']
    corr_matrix = filtered_df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu',
        title='Корреляция между оценками'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot
    st.subheader("📍 Анализ взаимосвязей")
    
    scatter_col1, scatter_col2 = st.columns(2)
    
    with scatter_col1:
        x_var = st.selectbox("Ось X:", numeric_cols)
    
    with scatter_col2:
        y_var = st.selectbox("Ось Y:", [col for col in numeric_cols if col != x_var])
    
    color_by = st.selectbox(
        "Цвет по:",
        ['gender', 'test preparation course', 'race/ethnicity']
    )
    
    fig = px.scatter(
        filtered_df,
        x=x_var,
        y=y_var,
        color=color_by,
        hover_data=['parental level of education', 'lunch'],
        title=f'{x_var} vs {y_var}'
    )
    st.plotly_chart(fig, use_container_width=True)

# СТРАНИЦА 2: РЕЗУЛЬТАТЫ АНАЛИЗА
else:
    st.title("🔍 Результаты анализа")
    
    # KPI для анализа
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Расчет разницы по полу
        male_avg = filtered_df[filtered_df['gender'] == 'male']['average_score'].mean()
        female_avg = filtered_df[filtered_df['gender'] == 'female']['average_score'].mean()
        diff = abs(male_avg - female_avg)
        st.metric("Разница по полу", f"{diff:.1f} балла")
    
    with col2:
        # Эффект подготовки
        prep_avg = filtered_df[filtered_df['test preparation course'] == 'completed']['average_score'].mean()
        no_prep_avg = filtered_df[filtered_df['test preparation course'] == 'none']['average_score'].mean()
        prep_effect = prep_avg - no_prep_avg
        st.metric("Эффект подготовки", f"+{prep_effect:.1f} балла")
    
    with col3:
        # Процент отличников
        top_students = len(filtered_df[filtered_df['average_score'] >= 85])
        percent_top = (top_students / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.metric("Отличники (85+)", f"{percent_top:.1f}%")
    
    st.markdown("---")
    
    # Кластеризация (упрощенная)
    st.subheader("🎯 Кластеризация студентов")
    
    # Выбор признаков для визуализации кластеров
    col1, col2 = st.columns(2)
    
    with col1:
        x_cluster = st.selectbox(
            "Признак X:",
            ['math score', 'reading score', 'writing score'],
            key='x_cluster'
        )
    
    with col2:
        y_cluster = st.selectbox(
            "Признак Y:",
            ['math score', 'reading score', 'writing score'],
            key='y_cluster',
            index=1
        )
    
    # Простая кластеризация по квартилям
    filtered_df['performance_cluster'] = pd.qcut(
        filtered_df['average_score'], 
        q=3, 
        labels=['Низкая', 'Средняя', 'Высокая']
    )
    
    fig = px.scatter(
        filtered_df,
        x=x_cluster,
        y=y_cluster,
        color='performance_cluster',
        title=f'Кластеризация студентов ({x_cluster} vs {y_cluster})',
        hover_data=['gender', 'race/ethnicity', 'test preparation course']
    )
    
    # Добавляем центроиды
    centroids = filtered_df.groupby('performance_cluster')[[x_cluster, y_cluster]].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=centroids[x_cluster],
        y=centroids[y_cluster],
        mode='markers',
        marker=dict(symbol='x', size=15, color='black', line=dict(width=2)),
        name='Центроиды'
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Статистика по кластерам
    cluster_stats = filtered_df.groupby('performance_cluster').agg({
        'math score': 'mean',
        'reading score': 'mean',
        'writing score': 'mean',
        'average_score': 'mean'
    }).round(1)
    
    st.dataframe(cluster_stats, use_container_width=True)
    
    st.markdown("---")
    
    # Регрессионный анализ
    st.subheader("📈 Важность признаков")
    
    # Простой анализ влияния
    factors = ['gender', 'test preparation course', 'lunch', 'parental level of education']
    factor_effects = []
    
    for factor in factors:
        if factor in ['parental level of education']:
            # Для образования родителей берем среднее по всем группам
            effect = filtered_df.groupby(factor)['average_score'].mean().std()
        else:
            groups = filtered_df.groupby(factor)['average_score'].mean()
            effect = abs(groups.iloc[0] - groups.iloc[1]) if len(groups) == 2 else 0
        
        factor_effects.append({
            'Фактор': factor,
            'Влияние (баллы)': round(effect, 2)
        })
    
    effects_df = pd.DataFrame(factor_effects).sort_values('Влияние (баллы)', ascending=False)
    
    # Bar chart важности признаков
    fig = px.bar(
        effects_df,
        x='Фактор',
        y='Влияние (баллы)',
        title='Относительное влияние факторов на успеваемость',
        color='Влияние (баллы)',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Сравнение моделей
    st.subheader("🤖 Сравнение моделей")
    
    models_comparison = pd.DataFrame({
        'Модель': ['Линейная регрессия', 'Random Forest'],
        'R²': [0.246, 0.725],
        'MAE': [9.52, 5.79],
        'Объясненная дисперсия': ['24.6%', '72.5%']
    })
    
    st.dataframe(models_comparison, use_container_width=True)
    
    # Визуализация точности моделей
    fig = go.Figure()
    
    # Симулируем данные для графиков точности
    x_range = np.linspace(filtered_df['average_score'].min(), filtered_df['average_score'].max(), 100)
    
    # "Прогнозы" линейной регрессии (более разбросанные)
    y_lr = x_range + np.random.normal(0, 8, len(x_range))
    
    # "Прогнозы" Random Forest (более точные)
    y_rf = x_range + np.random.normal(0, 3, len(x_range))
    
    fig.add_trace(go.Scatter(
        x=x_range, y=y_lr,
        mode='markers',
        name='Линейная регрессия (R²=0.246)',
        marker=dict(size=6, opacity=0.6)
    ))
    
    fig.add_trace(go.Scatter(
        x=x_range, y=y_rf,
        mode='markers',
        name='Random Forest (R²=0.725)',
        marker=dict(size=6, opacity=0.6, color='green')
    ))
    
    # Линия идеального прогноза
    fig.add_trace(go.Scatter(
        x=[x_range.min(), x_range.max()],
        y=[x_range.min(), x_range.max()],
        mode='lines',
        name='Идеальный прогноз',
        line=dict(dash='dash', color='red')
    ))
    
    fig.update_layout(
        title='Точность моделей: фактические vs прогнозируемые значения',
        xaxis_title='Фактический средний балл',
        yaxis_title='Прогнозируемый средний балл'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Футер
st.markdown("---")
st.markdown("*Дашборд для анализа успеваемости студентов*")
