# --- 라이브러리 임포트 ---
import streamlit as st
import pandas as pd
import numpy as np
import json
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px

# =========================
# 0. 유틸리티 및 통계 계산 함수
# =========================
def format_p_value(p):
    """p-값을 별(*) 형식으로 반환"""
    if p < 0.001:
        return f"{p:.3f} (***)"
    elif p < 0.01:
        return f"{p:.3f} (**)"
    elif p < 0.05:
        return f"{p:.3f} (*)"
    else:
        return f"{p:.3f}"

def calculate_paired_stats(pre_data, post_data):
    """대응표본 t-검정, Cohen's dz, 95% CI 계산"""
    temp_df = pd.DataFrame({'pre': pre_data, 'post': post_data}).dropna()
    pre, post = temp_df['pre'], temp_df['post']
    
    if len(pre) < 2:
        return {}

    # 대응표본 t-검정
    t_stat, p_value = stats.ttest_rel(pre, post)
    diff = post - pre
    n = len(diff)
    mean_diff = np.mean(diff)
    sd_diff = np.std(diff, ddof=1)

    cohen_dz = mean_diff / sd_diff if sd_diff != 0 else 0

    # 95% CI
    se_diff = sd_diff / np.sqrt(n)
    t_critical = stats.t.ppf(0.975, df=n-1)
    ci_low = mean_diff - t_critical * se_diff
    ci_high = mean_diff + t_critical * se_diff
    
    return {
        '사전 평균': np.mean(pre),
        '사후 평균': np.mean(post),
        '사전 표준편차': np.std(pre, ddof=1),
        '사후 표준편차': np.std(post, ddof=1),
        '평균 차이': mean_diff,
        't-값': t_stat,
        'p-값': p_value,
        "Cohen's dz": cohen_dz,
        '95% CI 하한': ci_low,
        '95% CI 상한': ci_high,
    }

# =========================
# 1. 앱 기본 설정
# =========================
st.set_page_config(layout="wide", page_title="대응표본 t-검정 분석기")
st.title("📄 교육연구대회용 사전-사후 데이터 분석")
st.subheader("대응표본 t-검정, 효과크기, 시각화 (문항/요인 단위 지원)")
st.write("---")

# =========================
# 2. 사이드바: 파일 업로드 및 옵션
# =========================
with st.sidebar:
    st.header("1. 데이터 업로드")
    uploaded_file = st.file_uploader("CSV 파일 업로드", type=['csv'])
    
    if uploaded_file:
        st.header("2. 분석 옵션")
        analysis_unit = st.radio(
            "분석 단위 선택",
            ('문항 단위', '요인 단위'),
            help="'문항 단위' → 개별 문항 비교, '요인 단위' → 여러 문항을 합산/평균"
        )

        # 요인 단위 추가 설정
        if analysis_unit == '요인 단위':
            agg_method = st.radio("요인 점수 집계 방식", ('평균(mean)', '합산(sum)'), horizontal=True)
            use_reverse = st.checkbox("역문항 처리")
            
            if use_reverse:
                max_score = st.number_input("리커트 척도 최대 점수", min_value=1, value=5, step=1)
                reverse_items_str = st.text_area("역문항 목록 (쉼표로 구분)", placeholder="예: 사전3, 사전5, 사후3, 사후5")

            st.subheader("요인-문항 매핑 입력")
            factor_map_str = st.text_area(
                "JSON 형식 입력",
                height=220,
                placeholder='''{
    "주도성": {
        "pre": ["사전1", "사전2", "사전3"],
        "post": ["사후1", "사후2", "사후3"]
    }
}'''
            )

# =========================
# 3. 메인 화면
# =========================
if uploaded_file:
    try:
        df_original = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        df = df_original.copy()
        st.header("📋 데이터 미리보기")
        st.dataframe(df.head())
        options = df.columns.tolist()
    except Exception as e:
        st.error(f"CSV 읽기 오류: {e}")
        st.stop()

    # --- 문항 단위 UI ---
    if analysis_unit == '문항 단위':
        st.header("🔍 분석 변수 선택 (문항 단위)")
        col1, col2 = st.columns(2)
        with col1:
            pre_vars = st.multiselect('사전 검사 변수', options, key="pre_item")
        with col2:
            post_vars = st.multiselect('사후 검사 변수', options, key="post_item")
        
        if pre_vars and post_vars and len(pre_vars) != len(post_vars):
            st.warning("⚠️ 사전-사후 변수 개수를 동일하게 선택하세요.")

    st.write("---")

    # =========================
    # 4. 분석 실행
    # =========================
    if st.button('🚀 분석 실행', type="primary"):
        results, df_for_plotting = [], pd.DataFrame()

        # --- 요인 단위 ---
        if analysis_unit == '요인 단위':
            try:
                if not factor_map_str:
                    st.error("요인-문항 매핑을 입력하세요.")
                    st.stop()
                factor_map = json.loads(factor_map_str)

                # (선택) 역문항 처리
                if use_reverse and reverse_items_str:
                    reverse_items = [item.strip() for item in reverse_items_str.split(',')]
                    for item in reverse_items:
                        if item in df.columns:
                            df[item] = (max_score + 1) - df[item]
                        else:
                            st.warning(f"⚠️ '{item}' 컬럼이 없어 역코딩 건너뜀")

                # 요인별 점수 계산 및 분석
                for factor, items in factor_map.items():
                    pre_cols, post_cols = items.get('pre', []), items.get('post', [])
                    if not pre_cols or not post_cols:
                        st.error(f"'{factor}' 요인에 pre/post 문항 없음")
                        continue
                    
                    factor_df_temp = df[pre_cols + post_cols].dropna()
                    if agg_method == '평균(mean)':
                        pre_score = factor_df_temp[pre_cols].mean(axis=1)
                        post_score = factor_df_temp[post_cols].mean(axis=1)
                    else:
                        pre_score = factor_df_temp[pre_cols].sum(axis=1)
                        post_score = factor_df_temp[post_cols].sum(axis=1)

                    stats_result = calculate_paired_stats(pre_score, post_score)
                    if stats_result:
                        stats_result['요인'] = factor
                        results.append(stats_result)
                        df_for_plotting[f"{factor}_사전"] = pre_score
                        df_for_plotting[f"{factor}_사후"] = post_score

            except json.JSONDecodeError:
                st.error("❌ JSON 형식 오류. 예시 참고 후 수정하세요.")

        # --- 문항 단위 ---
        else:
            if not pre_vars or not post_vars or len(pre_vars) != len(post_vars):
                st.error("사전-사후 변수를 올바르게 선택하세요.")
            else:
                for pre_var, post_var in zip(pre_vars, post_vars):
                    stats_result = calculate_paired_stats(df[pre_var], df[post_var])
                    if stats_result:
                        stats_result['사전 변수'] = pre_var
                        stats_result['사후 변수'] = post_var
                        results.append(stats_result)
                df_for_plotting = df

        # =========================
        # 5. 분석 결과 출력
        # =========================
        if results:
            st.header("📊 분석 결과")
            results_df = pd.DataFrame(results)

            # 컬럼 정리
            if analysis_unit == '요인 단위':
                cols_order = ['요인', '사전 평균', '사후 평균', '평균 차이',
                              '사전 표준편차', '사후 표준편차',
                              't-값', 'p-값', "Cohen's dz", '95% CI 하한', '95% CI 상한']
            else:
                cols_order = ['사전 변수', '사후 변수', '사전 평균', '사후 평균', '평균 차이',
                              '사전 표준편차', '사후 표준편차',
                              't-값', 'p-값', "Cohen's dz", '95% CI 하한', '95% CI 상한']
            
            results_df['유의성'] = results_df['p-값'].apply(format_p_value)
            results_df = results_df[[col for col in cols_order if col in results_df.columns] + ['유의성']]
            
            # 포맷팅
            fmt = {col: '{:.3f}' for col in results_df.columns if results_df[col].dtype == 'float64'}
            st.dataframe(results_df.style.format(fmt))

            # 다운로드
            csv_data = results_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button("📥 결과 CSV 다운로드", csv_data, "t-test_results.csv", "text/csv")
            st.info("p-값: * p<.05 / ** p<.01 / *** p<.001")

            # =========================
            # 6. 시각화
            # =========================
            st.header("📈 시각화")
            if analysis_unit == '요인 단위':
                plot_vars = [(f"{r['요인']}_사전", f"{r['요인']}_사후") for _, r in results_df.iterrows()]
            else:
                plot_vars = zip(pre_vars, post_vars)
            
            for pre_var, post_var in plot_vars:
                st.subheader(f"'{pre_var.replace('_사전','').replace('_사후','')}' 결과")

                # Raincloud Plot
                plot_df_long = pd.DataFrame({
                    '점수': pd.concat([df_for_plotting[pre_var], df_for_plotting[post_var]], ignore_index=True),
                    '시점': ['사전'] * len(df_for_plotting) + ['사후'] * len(df_for_plotting)
                })
                fig_rain = px.violin(plot_df_long, y='점수', x='시점', color='시점',
                                     box=True, points='all',
                                     color_discrete_map={'사전': 'blue', '사후': 'orange'})
                fig_rain.update_layout(title_text='사전-사후 분포 (Raincloud)', showlegend=False)
                st.plotly_chart(fig_rain, use_container_width=True)

                # Paired Line Plot
                fig_line = go.Figure()
                for i in range(len(df_for_plotting)):
                    fig_line.add_trace(go.Scatter(
                        x=['사전', '사후'],
                        y=[df_for_plotting.loc[i, pre_var], df_for_plotting.loc[i, post_var]],
                        mode='lines+markers',
                        line=dict(color='grey', width=1),
                        marker=dict(size=6),
                        showlegend=False
                    ))
                fig_line.update_layout(title_text='개별 변화 추이', xaxis_title='시점', yaxis_title='점수')
                st.plotly_chart(fig_line, use_container_width=True)
                st.write("---")

        else:
            st.warning("❗ 분석 결과가 없습니다. 데이터/설정을 확인하세요.")

else:
    st.info("👈 사이드바에서 CSV 파일을 업로드하세요.")

# =========================
# 7. Footer
# =========================
st.divider()
st.markdown("<div style='text-align:center; color:grey;'>© 2025 이대형. All rights reserved.</div>", unsafe_allow_html=True)
