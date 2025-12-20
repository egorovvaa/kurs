import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Настройка страницы
st.set_page_config(
    page_title="Анализ успеваемости студентов",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Загрузка данных
@st.cache_data
def load_data():
    df = pd.read_csv("StudentsPerformance.csv")
    # Добавляем новые признаки
    df['total_score'] = df['math score'] + df['reading score'] + df['writing score']
    df['average_score'] = df['total_score'] / 3
    
    # Категории успеваемости
    conditions = [
        (df['average_score'] >= 90),
        (df['average_score'] >= 80) & (df['average_score'] < 90),
        (df['average_score'] >= 70) & (df['average_score'] < 80),
        (df['average_score'] >= 60) & (df['average_score'] < 70),
        (df['average_score'] < 60)
    ]
    categories = ['Отличники', 'Хорошисты', 'Средние', 'Ниже среднего', 'Неуспевающие']
    df['performance_category'] = np.select(conditions, categories)
    
    return df

# Загрузка данных
df = load_data()

# Стили CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6B7280;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown('<h1 class="main-header">📊 Анализ успеваемости студентов</h1>', unsafe_allow_html=True)
st.markdown("---")

# Навигация
page = st.sidebar.selectbox("Выберите страницу", 
                           ["📈 Визуализация данных", 
                            "🔍 Результаты анализа",
                            "📋 О датасете"])

# Общие фильтры в сайдбаре
st.sidebar.markdown("### 🔍 Фильтры данных")

# Фильтры
gender_filter = st.sidebar.multiselect(
    "Пол:",
    options=df['gender'].unique(),
    default=df['gender'].unique()
)

ethnicity_filter = st.sidebar.multiselect(
    "Этническая группа:",
    options=df['race/ethnicity'].unique(),
    default=df['race/ethnicity'].unique()
)

prep_filter = st.sidebar.multiselect(
    "Подготовка к тесту:",
    options=df['test preparation course'].unique(),
    default=df['test preparation course'].unique()
)

# Фильтр по баллам
score_range = st.sidebar.slider(
    "Диапазон среднего балла:",
    min_value=float(df['average_score'].min()),
    max_value=float(df['average_score'].max()),
    value=(float(df['average_score'].min()), float(df['average_score'].max()))
)

# Применение фильтров
filtered_df = df[
    (df['gender'].isin(gender_filter)) &
    (df['race/ethnicity'].isin(ethnicity_filter)) &
    (df['test preparation course'].isin(prep_filter)) &
    (df['average_score'] >= score_range[0]) &
    (df['average_score'] <= score_range[1])
]

# СТРАНИЦА 1: ВИЗУАЛИЗАЦИЯ ДАННЫХ
if page == "📈 Визуализация данных":
    st.markdown('<h2 class="sub-header">Визуализация исходных данных</h2>', unsafe_allow_html=True)
    
    # KPI карточки
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Всего студентов</div>
        </div>
        """.format(len(filtered_df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.1f}</div>
            <div class="metric-label">Средний балл</div>
        </div>
        """.format(filtered_df['average_score'].mean()), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.1f}</div>
            <div class="metric-label">Медианный балл</div>
        </div>
        """.format(filtered_df['average_score'].median()), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{:.1f}</div>
            <div class="metric-label">Станд. отклонение</div>
        </div>
        """.format(filtered_df['average_score'].std()), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Таблица с данными
    st.markdown("### 📋 Таблица данных")
    st.dataframe(filtered_df, use_container_width=True, height=300)
    
    # Экспорт данных
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Скачать отфильтрованные данные",
        data=csv,
        file_name="filtered_student_data.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Распределение данных
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Распределение оценок")
        
        # Выбор предмета для гистограммы
        subject = st.selectbox(
            "Выберите предмет для анализа:",
            ['math score', 'reading score', 'writing score', 'total_score', 'average_score'],
            key='hist_subject'
        )
        
        fig = px.histogram(
            filtered_df, 
            x=subject,
            nbins=30,
            color='gender',
            barmode='overlay',
            title=f'Распределение {subject}',
            labels={subject: 'Баллы', 'count': 'Количество студентов'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Box plot по группам")
        
        group_by = st.selectbox(
            "Группировать по:",
            ['gender', 'race/ethnicity', 'parental level of education', 'test preparation course', 'lunch'],
            key='box_group'
        )
        
        fig = px.box(
            filtered_df,
            x=group_by,
            y='average_score',
            color=group_by,
            points='all',
            title=f'Средний балл по {group_by}',
            labels={'average_score': 'Средний балл', group_by: 'Группа'}
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Категориальные распределения
    st.markdown("### 📊 Категориальные распределения")
    
    cat_col1, cat_col2 = st.columns(2)
    
    with cat_col1:
        cat_var = st.selectbox(
            "Выберите категориальную переменную:",
            ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course', 'performance_category'],
            key='cat1'
        )
        
        fig = px.pie(
            filtered_df,
            names=cat_var,
            title=f'Распределение по {cat_var}',
            hole=0.3
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with cat_col2:
        # Bar chart сравнения
        comp_var = st.selectbox(
            "Сравнить по:",
            ['gender', 'race/ethnicity', 'parental level of education'],
            key='cat2'
        )
        
        avg_scores = filtered_df.groupby(comp_var)[['math score', 'reading score', 'writing score']].mean().reset_index()
        avg_scores_melted = avg_scores.melt(id_vars=[comp_var], 
                                           value_vars=['math score', 'reading score', 'writing score'],
                                           var_name='Предмет', 
                                           value_name='Средний балл')
        
        fig = px.bar(
            avg_scores_melted,
            x=comp_var,
            y='Средний балл',
            color='Предмет',
            barmode='group',
            title=f'Средние баллы по предметам ({comp_var})'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Корреляционная матрица
    st.markdown("### 🔗 Корреляционная матрица")
    
    numeric_cols = ['math score', 'reading score', 'writing score', 'total_score', 'average_score']
    corr_matrix = filtered_df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1,
        title='Корреляция между оценками',
        labels=dict(x="Признаки", y="Признаки", color="Корреляция")
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot для пар признаков
    st.markdown("### 📍 Scatter plot для анализа взаимосвязей")
    
    scatter_col1, scatter_col2 = st.columns(2)
    
    with scatter_col1:
        x_var = st.selectbox(
            "Выберите X:",
            numeric_cols,
            key='scatter_x'
        )
    
    with scatter_col2:
        y_var = st.selectbox(
            "Выберите Y:",
            [col for col in numeric_cols if col != x_var],
            key='scatter_y'
        )
    
    color_by = st.selectbox(
        "Цвет по:",
        ['gender', 'race/ethnicity', 'test preparation course', 'performance_category'],
        key='scatter_color'
    )
    
    fig = px.scatter(
        filtered_df,
        x=x_var,
        y=y_var,
        color=color_by,
        hover_data=['parental level of education', 'lunch'],
        title=f'{x_var} vs {y_var}',
        labels={x_var: x_var, y_var: y_var}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

# СТРАНИЦА 2: РЕЗУЛЬТАТЫ АНАЛИЗА
elif page == "🔍 Результаты анализа":
    st.markdown('<h2 class="sub-header">Результаты анализа</h2>', unsafe_allow_html=True)
    
    # KPI для анализа
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # T-test для гендерных различий
        male_scores = filtered_df[filtered_df['gender'] == 'male']['average_score']
        female_scores = filtered_df[filtered_df['gender'] == 'female']['average_score']
        if len(male_scores) > 0 and len(female_scores) > 0:
            t_stat, p_value = stats.ttest_ind(male_scores, female_scores, equal_var=False)
            sig_diff = "✓" if p_value < 0.05 else "✗"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{sig_diff}</div>
                <div class="metric-label">Различие по полу (p={p_value:.4f})</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # Влияние подготовки
        prep_scores = filtered_df[filtered_df['test preparation course'] == 'completed']['average_score']
        no_prep_scores = filtered_df[filtered_df['test preparation course'] == 'none']['average_score']
        if len(prep_scores) > 0 and len(no_prep_scores) > 0:
            prep_effect = prep_scores.mean() - no_prep_scores.mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">+{prep_effect:.1f}</div>
                <div class="metric-label">Влияние подготовки</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        # Процент отличников
        top_students = filtered_df[filtered_df['performance_category'] == 'Отличники']
        percent_top = (len(top_students) / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{percent_top:.1f}%</div>
            <div class="metric-label">Отличники</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Кластеризация студентов
    st.markdown("### 🎯 Кластеризация студентов по успеваемости")
    
    # Выбор признаков для кластеризации
    features_for_clustering = st.multiselect(
        "Выберите признаки для кластеризации:",
        ['math score', 'reading score', 'writing score'],
        default=['math score', 'reading score', 'writing score']
    )
    
    n_clusters = st.slider("Количество кластеров:", 2, 5, 3)
    
    if len(features_for_clustering) >= 2:
        # Подготовка данных
        X = filtered_df[features_for_clustering].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Кластеризация
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        filtered_df['cluster'] = clusters
        
        # Расчет silhouette score
        if n_clusters > 1:
            silhouette_avg = silhouette_score(X_scaled, clusters)
            st.info(f"Silhouette Score: {silhouette_avg:.3f}")
        
        # Визуализация кластеров
        col1, col2 = st.columns(2)
        
        with col1:
            # 2D scatter plot (первые два признака)
            if len(features_for_clustering) >= 2:
                fig = px.scatter(
                    filtered_df,
                    x=features_for_clustering[0],
                    y=features_for_clustering[1],
                    color='cluster',
                    title=f'Кластеризация студентов ({features_for_clustering[0]} vs {features_for_clustering[1]})',
                    hover_data=['gender', 'race/ethnicity', 'average_score'],
                    labels={'cluster': 'Кластер'}
                )
                # Добавление центроидов
                centroids = scaler.inverse_transform(kmeans.cluster_centers_)
                fig.add_trace(go.Scatter(
                    x=centroids[:, features_for_clustering.index(features_for_clustering[0])],
                    y=centroids[:, features_for_clustering.index(features_for_clustering[1])],
                    mode='markers',
                    marker=dict(symbol='x', size=15, color='black', line=dict(width=2)),
                    name='Центроиды'
                ))
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Средние показатели по кластерам
            cluster_stats = filtered_df.groupby('cluster').agg({
                'math score': 'mean',
                'reading score': 'mean',
                'writing score': 'mean',
                'average_score': 'mean',
                'gender': lambda x: (x == 'female').mean() * 100,
                'test preparation course': lambda x: (x == 'completed').mean() * 100
            }).round(1)
            
            cluster_stats.columns = ['Математика', 'Чтение', 'Письмо', 'Средний', '% Женщин', '% С подготовкой']
            st.dataframe(cluster_stats, use_container_width=True)
            
            # Интерпретация кластеров
            st.markdown("#### 📝 Интерпретация кластеров:")
            best_cluster = cluster_stats['Средний'].idxmax()
            worst_cluster = cluster_stats['Средний'].idxmin()
            st.write(f"**Кластер {best_cluster}**: студенты с наивысшими показателями")
            st.write(f"**Кластер {worst_cluster}**: студенты с низкими показателями")
    
    st.markdown("---")
    
    # Анализ влияния факторов
    st.markdown("### 📈 Анализ влияния факторов на успеваемость")
    
    # Влияние уровня образования родителей
    st.markdown("#### 🎓 Влияние уровня образования родителей")
    
    education_order = ['some high school', 'high school', 'some college', 
                      "associate's degree", "bachelor's degree", "master's degree"]
    
    # Присваиваем порядковые номера
    edu_rank = {edu: i for i, edu in enumerate(education_order)}
    filtered_df['edu_encoded'] = filtered_df['parental level of education'].map(edu_rank)
    
    # Корреляция
    if len(filtered_df) > 1:
        corr, p_val = stats.spearmanr(filtered_df['edu_encoded'], filtered_df['average_score'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart средних баллов по уровню образования
            edu_means = filtered_df.groupby('parental level of education')['average_score']\
                .mean().reindex(education_order)
            
            fig = px.bar(
                x=edu_means.index,
                y=edu_means.values,
                title='Средний балл по уровню образования родителей',
                labels={'x': 'Уровень образования', 'y': 'Средний балл'},
                color=edu_means.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400, xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric(
                "Корреляция с уровнем образования",
                f"{corr:.3f}",
                delta=f"p-value: {p_val:.4f}" if p_val < 0.05 else "Незначимо"
            )
            
            # Дополнительная статистика
            st.markdown("#### 📊 Статистика по факторам:")
            
            factor_stats = []
            for factor in ['lunch', 'test preparation course']:
                for value in filtered_df[factor].unique():
                    subset = filtered_df[filtered_df[factor] == value]
                    factor_stats.append({
                        'Фактор': f"{factor} - {value}",
                        'Средний балл': subset['average_score'].mean(),
                        'Количество': len(subset)
                    })
            
            factor_df = pd.DataFrame(factor_stats)
            st.dataframe(factor_df, use_container_width=True)
    
    st.markdown("---")
    
    # Прогноз успеваемости
    st.markdown("### 🔮 Прогноз успеваемости")
    
    # Интерактивный калькулятор
    st.markdown("#### 🧮 Калькулятор прогноза успеваемости")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        math_score = st.slider("Балл по математике:", 0, 100, 70)
        gender = st.selectbox("Пол:", ['male', 'female'])
    
    with col2:
        reading_score = st.slider("Балл по чтению:", 0, 100, 70)
        ethnicity = st.selectbox("Этническая группа:", df['race/ethnicity'].unique())
    
    with col3:
        writing_score = st.slider("Балл по письму:", 0, 100, 70)
        prep = st.selectbox("Подготовка:", ['none', 'completed'])
    
    # Расчет прогноза
    total_pred = math_score + reading_score + writing_score
    avg_pred = total_pred / 3
    
    # Определение категории
    if avg_pred >= 90:
        category = "Отличник"
    elif avg_pred >= 80:
        category = "Хорошист"
    elif avg_pred >= 70:
        category = "Средний"
    elif avg_pred >= 60:
        category = "Ниже среднего"
    else:
        category = "Неуспевающий"
    
    # Отображение результатов
    st.markdown("---")
    result_col1, result_col2, result_col3 = st.columns(3)
    
    with result_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_pred}</div>
            <div class="metric-label">Общий балл</div>
        </div>
        """, unsafe_allow_html=True)
    
    with result_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_pred:.1f}</div>
            <div class="metric-label">Средний балл</div>
        </div>
        """, unsafe_allow_html=True)
    
    with result_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{category}</div>
            <div class="metric-label">Категория</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Сравнение с реальными данными
    st.markdown("#### 📊 Сравнение с реальными данными:")
    
    similar_students = df[
        (df['math score'].between(math_score-10, math_score+10)) &
        (df['reading score'].between(reading_score-10, reading_score+10)) &
        (df['writing score'].between(writing_score-10, writing_score+10))
    ]
    
    if len(similar_students) > 0:
        avg_similar = similar_students['average_score'].mean()
        st.write(f"Средний балл похожих студентов: **{avg_similar:.1f}**")
        st.write(f"Количество похожих студентов: **{len(similar_students)}**")
    else:
        st.info("Похожих студентов в датасете не найдено")

# СТРАНИЦА 3: О ДАТАСЕТЕ
else:
    st.markdown('<h2 class="sub-header">О датасете</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📋 Описание датасета
        
        Датасет содержит информацию об успеваемости 1000 студентов по трем предметам:
        математике, чтению и письму.
        
        ### 📊 Структура данных
        
        **Признаки:**
        1. **gender** - пол студента (male/female)
        2. **race/ethnicity** - этническая группа (группы A-E)
        3. **parental level of education** - уровень образования родителей
        4. **lunch** - тип питания (standard/free/reduced)
        5. **test preparation course** - прохождение подготовительного курса
        
        **Целевые переменные:**
        1. **math score** - балл по математике (0-100)
        2. **reading score** - балл по чтению (0-100)
        3. **writing score** - балл по письму (0-100)
        
        ### 🎯 Цель анализа
        
        Выявление факторов, влияющих на успеваемость студентов, и создание модели
        для прогнозирования результатов.
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Основные показатели
        
        **Общая статистика:**
        """)
        
        stats_data = {
            'Показатель': ['Всего записей', 'Колонок', 'Пропусков', 'Дубликатов'],
            'Значение': [len(df), len(df.columns), df.isnull().sum().sum(), df.duplicated().sum()]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        st.markdown("""
        ### 📁 Информация о колонках
        """)
        
        column_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique = df[col].nunique()
            column_info.append([col, dtype, unique])
        
        columns_df = pd.DataFrame(column_info, columns=['Колонка', 'Тип', 'Уникальных'])
        st.dataframe(columns_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Примеры данных
    st.markdown("### 📝 Примеры данных")
    
    tab1, tab2, tab3 = st.tabs(["Первые 10 строк", "Случайная выборка", "Статистика"])
    
    with tab1:
        st.dataframe(df.head(10), use_container_width=True)
    
    with tab2:
        st.dataframe(df.sample(10, random_state=42), use_container_width=True)
    
    with tab3:
        st.dataframe(df.describe(), use_container_width=True)
    
    # Инструкции
    st.markdown("---")
    st.markdown("""
    ### 🚀 Инструкции по использованию
    
    1. **Навигация**: Используйте сайдбар для переключения между страницами
    2. **Фильтры**: Настройте фильтры в сайдбаре для анализа подгрупп
    3. **Интерактивность**: Наводите курсор на графики для детальной информации
    4. **Экспорт**: Скачивайте отфильтрованные данные в формате CSV
    5. **Анализ**: Исследуйте влияния различных факторов на успеваемость
    """)

# Футер
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.9rem;">
    <p>📊 Курсовая работа по анализу данных | Анализ успеваемости студентов</p>
    <p>Дашборд создан с использованием Streamlit</p>
</div>
""", unsafe_allow_html=True)
