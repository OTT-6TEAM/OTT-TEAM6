"""
TMDB TV Series Data Analysis Example
수집된 데이터를 분석하는 예제 스크립트
"""

import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path='tmdb_tv_series_data.parquet'):
    """Parquet 파일 로드"""
    print(f"Loading data from {file_path}...")
    df = pd.read_parquet(file_path)
    print(f"✓ Loaded {len(df)} TV series\n")
    return df


def basic_info(df):
    """기본 정보 출력"""
    print("="*60)
    print("BASIC INFORMATION")
    print("="*60)
    print(f"Total TV Series: {len(df)}")
    print(f"Date Range: {df['first_air_date'].min()} to {df['first_air_date'].max()}")
    print(f"Average Vote: {df['vote_average'].mean():.2f}")
    print(f"Average Vote Count: {df['vote_count'].mean():.0f}")
    print(f"\nColumns: {list(df.columns)}")
    print()


def analyze_producers(df):
    """총괄 프로듀서 분석"""
    print("="*60)
    print("EXECUTIVE PRODUCERS ANALYSIS")
    print("="*60)
    
    # 프로듀서 정보가 있는 시리즈
    df_with_producers = df[df['executive_producer_name'] != '']
    print(f"Series with Executive Producers: {len(df_with_producers)} ({len(df_with_producers)/len(df)*100:.1f}%)")
    
    # 모든 프로듀서 추출
    all_producers = []
    for names in df_with_producers['executive_producer_name']:
        if names:
            all_producers.extend([n.strip() for n in names.split(';')])
    
    producer_counts = pd.Series(all_producers).value_counts()
    print(f"\nTotal Unique Executive Producers: {len(producer_counts)}")
    print(f"\nTop 10 Most Frequent Executive Producers:")
    print(producer_counts.head(10))
    print()


def analyze_writers(df):
    """작가 분석"""
    print("="*60)
    print("WRITERS ANALYSIS")
    print("="*60)
    
    # 작가 정보가 있는 시리즈
    df_with_writers = df[df['writers_name'] != '']
    print(f"Series with Writers: {len(df_with_writers)} ({len(df_with_writers)/len(df)*100:.1f}%)")
    
    # 모든 작가 추출
    all_writers = []
    all_roles = []
    for idx, row in df_with_writers.iterrows():
        if row['writers_name']:
            names = [n.strip() for n in row['writers_name'].split(';')]
            roles = [r.strip() for r in row['writer_roles'].split(';')]
            all_writers.extend(names)
            all_roles.extend(roles)
    
    writer_counts = pd.Series(all_writers).value_counts()
    role_counts = pd.Series(all_roles).value_counts()
    
    print(f"\nTotal Unique Writers: {len(writer_counts)}")
    print(f"\nTop 10 Most Prolific Writers:")
    print(writer_counts.head(10))
    print(f"\nWriter Role Distribution:")
    print(role_counts)
    print()


def analyze_cast(df):
    """주연 배우 분석"""
    print("="*60)
    print("TOP CAST ANALYSIS")
    print("="*60)
    
    # 주연 배우 정보가 있는 시리즈
    df_with_cast = df[df['top_cast'] != '']
    print(f"Series with Top Cast: {len(df_with_cast)} ({len(df_with_cast)/len(df)*100:.1f}%)")
    
    # 5명 모두 있는 시리즈
    df_full_cast = df_with_cast[df_with_cast['top_cast'].str.split(';').str.len() == 5]
    print(f"Series with Full 5 Top Cast: {len(df_full_cast)} ({len(df_full_cast)/len(df)*100:.1f}%)")
    
    # 모든 배우 추출
    all_actors = []
    for names in df_with_cast['top_cast']:
        if names:
            all_actors.extend([n.strip() for n in names.split(';')])
    
    actor_counts = pd.Series(all_actors).value_counts()
    print(f"\nTotal Unique Top Cast Members: {len(actor_counts)}")
    print(f"\nTop 10 Most Frequent Top Cast Members:")
    print(actor_counts.head(10))
    
    # 성별 분포
    all_genders = []
    for genders in df_with_cast['top_cast_gender']:
        if genders:
            all_genders.extend([g.strip() for g in genders.split(';')])
    
    gender_map = {'0': 'Unknown', '1': 'Female', '2': 'Male'}
    gender_counts = pd.Series(all_genders).map(gender_map).value_counts()
    print(f"\nGender Distribution in Top Cast:")
    print(gender_counts)
    print()


def year_trends(df):
    """연도별 트렌드 분석"""
    print("="*60)
    print("YEAR TRENDS")
    print("="*60)
    
    # first_air_date에서 연도 추출
    df['year'] = pd.to_datetime(df['first_air_date'], errors='coerce').dt.year
    
    yearly_counts = df.groupby('year').size()
    print(f"\nTV Series per Year (Top 10):")
    print(yearly_counts.sort_values(ascending=False).head(10))
    
    # 연도별 평균 평점
    yearly_avg_vote = df.groupby('year')['vote_average'].mean()
    print(f"\nAverage Vote by Year (Recent 5 years):")
    print(yearly_avg_vote.tail(5))
    print()


def top_rated_series(df):
    """최고 평점 시리즈"""
    print("="*60)
    print("TOP RATED TV SERIES")
    print("="*60)
    
    # vote_count가 많은 것 중 평점이 높은 것
    df_popular = df[df['vote_count'] >= 100].sort_values('vote_average', ascending=False)
    
    print("\nTop 10 Highest Rated (with at least 100 votes):")
    for idx, row in df_popular.head(10).iterrows():
        print(f"{row['title']:40s} | Rating: {row['vote_average']:.1f} | Votes: {row['vote_count']:>5.0f} | Year: {row['first_air_date'][:4]}")
    print()


def missing_data_analysis(df):
    """결측값 분석"""
    print("="*60)
    print("MISSING DATA ANALYSIS")
    print("="*60)
    
    print("\nEmpty String Counts:")
    for col in df.columns:
        if df[col].dtype == 'object':
            empty_count = (df[col] == '').sum()
            if empty_count > 0:
                print(f"{col:40s}: {empty_count:>6d} ({empty_count/len(df)*100:>5.1f}%)")
    print()


def export_samples(df, n=100):
    """샘플 데이터 CSV로 내보내기"""
    print("="*60)
    print("EXPORTING SAMPLES")
    print("="*60)
    
    sample_file = f'tmdb_sample_{n}.csv'
    df.head(n).to_csv(sample_file, index=False)
    print(f"✓ Exported {n} samples to {sample_file}")
    print()


def main():
    """메인 분석 실행"""
    # 데이터 로드
    df = load_data()
    
    # 각종 분석 실행
    basic_info(df)
    analyze_producers(df)
    analyze_writers(df)
    analyze_cast(df)
    year_trends(df)
    top_rated_series(df)
    missing_data_analysis(df)
    
    # 샘플 데이터 내보내기
    export_samples(df, n=100)
    
    print("="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
