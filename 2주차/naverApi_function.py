from enum import Enum
import requests
import pandas as pd
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from tqdm.auto import tqdm

from bokeh.models import DatetimeTickFormatter
from bokeh.plotting import figure, show, output_notebook, save
from bokeh.models import ColumnDataSource, HoverTool, BoxAnnotation
from bokeh.palettes import Category10
from bokeh.layouts import layout
from bokeh.io.export import export_png

__all__ = [
    'TimeUnit',
    'APIEndpoint',
    'NaverAPIConfig',
    'CategoryRequestConfig',
    'DeviceRequestConfig',
    'GenderRequestConfig',
    'AgeRequestConfig',
    'KeywordRequestConfig',
    'NaverShoppingInsight'
]


class TimeUnit(Enum):
    """API 시간 단위 설정"""
    DATE = "date"
    WEEK = "week"
    MONTH = "month"


class APIEndpoint(Enum):
    """네이버 데이터랩 쇼핑 API 엔드포인트"""
    CATEGORIES = "https://openapi.naver.com/v1/datalab/shopping/categories"
    DEVICE = "https://openapi.naver.com/v1/datalab/shopping/category/device"
    GENDER = "https://openapi.naver.com/v1/datalab/shopping/category/gender"
    AGE = "https://openapi.naver.com/v1/datalab/shopping/category/age"
    KEYWORDS = "https://openapi.naver.com/v1/datalab/shopping/category/keywords"
    KEYWORD_DEVICE = "https://openapi.naver.com/v1/datalab/shopping/category/keyword/device"
    KEYWORD_GENDER = "https://openapi.naver.com/v1/datalab/shopping/category/keyword/gender"
    KEYWORD_AGE = "https://openapi.naver.com/v1/datalab/shopping/category/keyword/age"


class NaverAPIConfig:
    """네이버 API 인증 설정"""

    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret

    def get_headers(self) -> Dict[str, str]:
        return {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret,
            "Content-Type": "application/json"
        }


class BaseRequestConfig:
    """기본 요청 설정"""

    def __init__(
        self,
        start_date: str,
        end_date: str,
        category: str,
        time_unit: TimeUnit = TimeUnit.MONTH
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.category = category
        self.time_unit = time_unit

    def validate_dates(self) -> None:
        """날짜 형식 검증"""
        try:
            datetime.strptime(self.start_date, "%Y-%m-%d")
            datetime.strptime(self.end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"잘못된 날짜 형식: {str(e)}")


class CategoryRequestConfig:
    """카테고리 분석 요청 설정"""

    def __init__(
        self,
        start_date: str,
        end_date: str,
        categories: List[Dict[str, Any]],
        ages: List[str],
        category: str = "",
        time_unit: TimeUnit = TimeUnit.MONTH
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.categories = categories
        self.ages = ages
        self.category = category
        self.time_unit = time_unit

    def validate_dates(self) -> None:
        """날짜 형식 검증"""
        try:
            datetime.strptime(self.start_date, "%Y-%m-%d")
            datetime.strptime(self.end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"잘못된 날짜 형식: {str(e)}")

    def to_request_body(self) -> Dict[str, Any]:
        self.validate_dates()
        return {
            "startDate": self.start_date,
            "endDate": self.end_date,
            "timeUnit": self.time_unit.value,
            "category": self.categories,
            "ages": self.ages
        }


class DeviceRequestConfig:
    """디바이스별 분석 요청 설정"""

    def __init__(
        self,
        start_date: str,
        end_date: str,
        category: str,
        gender: str,
        ages: List[str],
        time_unit: TimeUnit = TimeUnit.MONTH
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.category = category
        self.gender = gender
        self.ages = ages
        self.time_unit = time_unit

    def validate_dates(self) -> None:
        """날짜 형식 검증"""
        try:
            datetime.strptime(self.start_date, "%Y-%m-%d")
            datetime.strptime(self.end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"잘못된 날짜 형식: {str(e)}")

    def to_request_body(self) -> Dict[str, Any]:
        self.validate_dates()
        return {
            "startDate": self.start_date,
            "endDate": self.end_date,
            "timeUnit": self.time_unit.value,
            "category": self.category,
            "gender": self.gender,
            "ages": self.ages
        }


class GenderRequestConfig:
    """성별 분석 요청 설정"""

    def __init__(
        self,
        start_date: str,
        end_date: str,
        category: str,
        ages: List[str],
        time_unit: TimeUnit = TimeUnit.MONTH
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.category = category
        self.ages = ages
        self.time_unit = time_unit

    def validate_dates(self) -> None:
        """날짜 형식 검증"""
        try:
            datetime.strptime(self.start_date, "%Y-%m-%d")
            datetime.strptime(self.end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"잘못된 날짜 형식: {str(e)}")

    def to_request_body(self) -> Dict[str, Any]:
        self.validate_dates()
        return {
            "startDate": self.start_date,
            "endDate": self.end_date,
            "timeUnit": self.time_unit.value,
            "category": self.category,
            "ages": self.ages
        }


class AgeRequestConfig:
    """연령대별 분석 요청 설정"""

    def __init__(
        self,
        start_date: str,
        end_date: str,
        category: str,
        ages: List[str],
        time_unit: TimeUnit = TimeUnit.MONTH
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.category = category
        self.ages = ages
        self.time_unit = time_unit

    def validate_dates(self) -> None:
        """날짜 형식 검증"""
        try:
            datetime.strptime(self.start_date, "%Y-%m-%d")
            datetime.strptime(self.end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"잘못된 날짜 형식: {str(e)}")

    def to_request_body(self) -> Dict[str, Any]:
        self.validate_dates()
        return {
            "startDate": self.start_date,
            "endDate": self.end_date,
            "timeUnit": self.time_unit.value,
            "category": self.category,
            "ages": self.ages
        }


class KeywordRequestConfig:
    """키워드 분석 요청 설정"""

    def __init__(
        self,
        start_date: str,
        end_date: str,
        category: str,
        keyword: Union[str, List[Dict[str, Any]]],
        time_unit: TimeUnit = TimeUnit.MONTH,
        ages: List[str] = None,
        device: str = "",
        gender: str = ""
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.category = category
        self.keyword = keyword
        self.time_unit = time_unit
        self.ages = ages if ages is not None else []
        self.device = device
        self.gender = gender

    def validate_dates(self) -> None:
        """날짜 형식 검증"""
        try:
            datetime.strptime(self.start_date, "%Y-%m-%d")
            datetime.strptime(self.end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"잘못된 날짜 형식: {str(e)}")

    def to_request_body(self) -> Dict[str, Any]:
        self.validate_dates()
        return {
            "startDate": self.start_date,
            "endDate": self.end_date,
            "timeUnit": self.time_unit.value,
            "category": self.category,
            "keyword": self.keyword,
            "device": self.device,
            "gender": self.gender,
            "ages": self.ages
        }


class NaverShoppingInsight:
    """네이버 쇼핑 인사이트 API 클래스"""

    DEVICE_GROUPS = ['pc', 'mobile']
    GENDER_GROUPS = {'f': 'female', 'm': 'male'}
    AGE_GROUPS = {
        '10': '10대', '20': '20대', '30': '30대',
        '40': '40대', '50': '50대', '60': '60대'
    }

    def __init__(self, api_config: NaverAPIConfig):
        self.api_config = api_config

    def _make_request(self, endpoint: APIEndpoint, body: Dict[str, Any]) -> Dict[str, Any]:
        """API 요청 실행"""
        try:
            response = requests.post(
                endpoint.value,
                headers=self.api_config.get_headers(),
                json=body
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API 요청 실패: {str(e)}")

    def fetch_shopping_trends(self, config: CategoryRequestConfig) -> pd.DataFrame:
        """쇼핑 트렌드 데이터 조회"""
        try:
            result = self._make_request(
                APIEndpoint.CATEGORIES, config.to_request_body())

            df = pd.DataFrame(result['results'][0]['data'])[['period']]

            for category_result in result['results']:
                category_data = pd.DataFrame(category_result['data'])
                df = df.merge(
                    category_data.rename(
                        columns={'ratio': category_result['title']}),
                    how='left',
                    on='period'
                )

            df['period'] = pd.to_datetime(df['period'])
            return df

        except (KeyError, IndexError) as e:
            raise Exception(f"데이터 파싱 실패: {str(e)}")

    def fetch_device_traffic(self, config: DeviceRequestConfig) -> pd.DataFrame:
        """디바이스별 트래픽 데이터 조회"""
        try:
            result = self._make_request(
                APIEndpoint.DEVICE, config.to_request_body())

            traffic_data = {
                'period': [],
                'pc': [],
                'mobile': []
            }

            for item in result['results'][0]['data']:
                if item['group'] == 'pc':
                    traffic_data['period'].append(item['period'])
                    traffic_data['pc'].append(item['ratio'])
                else:
                    traffic_data['mobile'].append(item['ratio'])

            df = pd.DataFrame(traffic_data)
            df['period'] = pd.to_datetime(df['period'])
            return df

        except (KeyError, IndexError) as e:
            raise Exception(f"데이터 파싱 실패: {str(e)}")

    def fetch_gender_traffic(self, config: GenderRequestConfig) -> pd.DataFrame:
        """성별 트래픽 데이터 조회"""
        try:
            result = self._make_request(
                APIEndpoint.GENDER, config.to_request_body())

            traffic_data = {
                'period': [],
                'female': [],
                'male': []
            }

            for item in result['results'][0]['data']:
                if item['group'] == 'f':
                    traffic_data['period'].append(item['period'])
                    traffic_data['female'].append(item['ratio'])
                else:
                    traffic_data['male'].append(item['ratio'])

            df = pd.DataFrame(traffic_data)
            df['period'] = pd.to_datetime(df['period'])
            return df

        except (KeyError, IndexError) as e:
            raise Exception(f"데이터 파싱 실패: {str(e)}")

    def fetch_age_traffic(self, config: AgeRequestConfig) -> pd.DataFrame:
        """연령대별 트래픽 데이터 조회"""
        try:
            result = self._make_request(
                APIEndpoint.AGE, config.to_request_body())

            all_dates = sorted(list(set(
                data['period'] for data in result['results'][0]['data']
            )))

            traffic_data = {
                'period': all_dates,
                **{age_label: [] for age_label in self.AGE_GROUPS.values()}
            }

            for date in all_dates:
                date_data = {age: None for age in self.AGE_GROUPS.keys()}

                for data in result['results'][0]['data']:
                    if data['period'] == date:
                        date_data[data['group']] = data['ratio']

                for age, label in self.AGE_GROUPS.items():
                    traffic_data[label].append(date_data[age] or '')

            df = pd.DataFrame(traffic_data)
            df['period'] = pd.to_datetime(df['period'])
            return df

        except (KeyError, IndexError) as e:
            raise Exception(f"데이터 파싱 실패: {str(e)}")

    def fetch_keyword_trends(self, config: KeywordRequestConfig) -> pd.DataFrame:
        """키워드 트렌드 데이터 조회"""
        try:
            result = self._make_request(
                APIEndpoint.KEYWORDS, config.to_request_body())

            df = pd.DataFrame(result['results'][0]['data'])[['period']]

            for keyword_result in result['results']:
                keyword_data = pd.DataFrame(keyword_result['data'])
                df = df.merge(
                    keyword_data.rename(
                        columns={'ratio': keyword_result['title']}),
                    how='left',
                    on='period'
                )

            df['period'] = pd.to_datetime(df['period'])
            return df

        except (KeyError, IndexError) as e:
            raise Exception(f"데이터 파싱 실패: {str(e)}")

    def fetch_keyword_device_traffic(self, config: KeywordRequestConfig) -> pd.DataFrame:
        """키워드별 디바이스 트래픽 데이터 조회"""
        try:
            result = self._make_request(
                APIEndpoint.KEYWORD_DEVICE, config.to_request_body())

            traffic_data = {
                'period': [],
                'pc': [],
                'mobile': []
            }

            for item in result['results'][0]['data']:
                if item['group'] == 'pc':
                    traffic_data['period'].append(item['period'])
                    traffic_data['pc'].append(item['ratio'])
                else:
                    traffic_data['mobile'].append(item['ratio'])

            df = pd.DataFrame(traffic_data)
            df['period'] = pd.to_datetime(df['period'])
            return df

        except (KeyError, IndexError) as e:
            raise Exception(f"데이터 파싱 실패: {str(e)}")

    def fetch_keyword_gender_traffic(self, config: KeywordRequestConfig) -> pd.DataFrame:
        """키워드별 성별 트래픽 데이터 조회"""
        try:
            result = self._make_request(
                APIEndpoint.KEYWORD_GENDER, config.to_request_body())

            traffic_data = {
                'period': [],
                'female': [],
                'male': []
            }

            for item in result['results'][0]['data']:
                if item['group'] == 'f':
                    traffic_data['period'].append(item['period'])
                    traffic_data['female'].append(item['ratio'])
                else:
                    traffic_data['male'].append(item['ratio'])

            df = pd.DataFrame(traffic_data)
            df['period'] = pd.to_datetime(df['period'])
            return df

        except (KeyError, IndexError) as e:
            raise Exception(f"데이터 파싱 실패: {str(e)}")

    def fetch_keyword_age_traffic(self, config: KeywordRequestConfig) -> pd.DataFrame:
        """키워드별 연령대 트래픽 데이터 조회"""
        try:
            result = self._make_request(
                APIEndpoint.KEYWORD_AGE, config.to_request_body())

            all_dates = sorted(list(set(
                data['period'] for data in result['results'][0]['data']
            )))

            traffic_data = {
                'period': all_dates,
                **{age_label: [] for age_label in self.AGE_GROUPS.values()}
            }

            for date in all_dates:
                date_data = {age: None for age in self.AGE_GROUPS.keys()}

                for data in result['results'][0]['data']:
                    if data['period'] == date:
                        date_data[data['group']] = data['ratio']

                for age, label in self.AGE_GROUPS.items():
                    traffic_data[label].append(date_data[age] or 0)

            df = pd.DataFrame(traffic_data)
            df['period'] = pd.to_datetime(df['period'])
            return df

        except (KeyError, IndexError) as e:
            raise Exception(f"데이터 파싱 실패: {str(e)}")


def generate_monthly_dates(start_date, end_date):
    """시작일부터 종료일까지 1달 단위로 날짜 구간 생성"""
    from datetime import datetime, timedelta
    from dateutil.relativedelta import relativedelta

    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    date_ranges = []
    current = start

    while current < end:
        next_month = current + relativedelta(months=1)
        if next_month > end:
            next_month = end

        date_ranges.append({
            'start': current.strftime('%Y-%m-%d'),
            'end': (next_month - timedelta(days=1)).strftime('%Y-%m-%d')
        })
        current = next_month

    return date_ranges


def plot_keyword_age_trends(df, word):
    output_notebook()

    p = figure(
        title=f"네이버 쇼핑 카테고리별 검색 클릭 비율 ({word})",
        x_axis_type="datetime",
        height=600,
        width=1200,
        tools="pan,box_zoom,reset,save",
        sizing_mode="stretch_width"
    )

    palette = Category10[10]

    # 연령대 레이블 매핑 - 실제 데이터프레임에 있는 컬럼만 포함
    age_labels = {}
    for col in df.columns[1:]:  # period 컬럼 제외
        if col in ['20', '30', '40']:  # 실제 존재하는 연령대 컬럼만 매핑
            age_labels[col] = f'{col}대'

    # df의 period 컬럼을 x축으로 사용
    for i, column in enumerate(sorted(df.columns[1:])):
        source = ColumnDataSource(
            df[['period', column]].rename(columns={column: 'ratio'}))
        age_label = age_labels.get(column, column)

        # ColumnDataSource에 legend_label 필드 추가
        source.data['legend_label'] = [age_label] * len(source.data['period'])

        p.line(x='period', y='ratio', legend_label=age_label,
               line_width=2, color=palette[i % 10], source=source)
        p.scatter(x='period', y='ratio', size=5,
                  color=palette[i % 10], source=source)

    hover = HoverTool(tooltips=[
        ('연령', '@legend_label'),
        ('비율', '@ratio{0.00}%'),
        ('일자', '@period{%F}')
    ], formatters={'@period': 'datetime'})

    p.add_tools(hover)
    p.xaxis.major_label_orientation = 0.8
    p.legend.click_policy = "hide"
    p.legend.location = "top_left"
    p.xaxis.axis_label = '날짜'
    p.yaxis.axis_label = '검색 비율(%)'

    show(p)

# 월별 데이터 수집 및 병합


def fetch_monthly_data(keywords, start_date, end_date):
    merged_df = pd.DataFrame()
    date_ranges = generate_monthly_dates(start_date, end_date)

    for date_range in tqdm(date_ranges, total=len(date_ranges)):
        # print(f"수집 중: {date_range['start']} ~ {date_range['end']}")
        # API 요청 설정
        category_config = KeywordRequestConfig(
            start_date=date_range['start'],
            end_date=date_range['end'],
            category="50000006",
            keyword=keywords,
            time_unit=TimeUnit.MONTH,
            ages=["20", "30", "40", "50", "60"]
        )

        # API 요청
        temp = client.fetch_keyword_age_traffic(category_config)
        merged_df = pd.concat([merged_df, temp])

    return merged_df


def main():
    # API 설정
    api_config = NaverAPIConfig(
        client_id="YOUR_CLIENT_ID",
        client_secret="YOUR_CLIENT_SECRET"
    )

    # API 클라이언트 생성
    client = NaverShoppingInsight(api_config)

    try:
        # 카테고리 분석
        category_config = CategoryRequestConfig(
            start_date="2025-01-01",
            end_date="2025-03-15",
            time_unit=TimeUnit.MONTH,
            categories=[
                {"name": "농산물", "param": ["50000160"]},
                {"name": "수산물", "param": ["50000159"]}
            ],
            ages=["20", "30", "40"]
        )
        df_trends = client.fetch_shopping_trends(category_config)
        print("카테고리 분석 결과:")
        print(df_trends)

        # 디바이스별 분석
        device_config = DeviceRequestConfig(
            start_date="2025-01-01",
            end_date="2025-03-15",
            category="50000006",
            gender="f",
            ages=["20", "30", "40"]
        )
        df_device = client.fetch_device_traffic(device_config)
        print("\n디바이스별 분석 결과:")
        print(df_device)

        # 성별 분석
        gender_config = GenderRequestConfig(
            start_date="2025-01-01",
            end_date="2025-03-15",
            category="50000006",
            ages=["20", "30", "40"]
        )
        df_gender = client.fetch_gender_traffic(gender_config)
        print("\n성별 분석 결과:")
        print(df_gender)

        # 연령대별 분석
        age_config = AgeRequestConfig(
            start_date="2023-01-01",
            end_date="2025-04-01",
            category="50000006"
        )
        df_age = client.fetch_age_traffic(age_config)
        print("\n연령대별 분석 결과:")
        print(df_age)

        # 키워드 트렌드 분석
        keyword_config = KeywordRequestConfig(
            start_date="2024-01-01",
            end_date="2024-04-01",
            category="50000006",
            keyword=[
                {"name": "식품/축산물", "param": ["축산물"]},
                {"name": "식품/수산물", "param": ["수산물"]},
                {"name": "식품/농산물", "param": ["농산물"]}
            ]
        )
        df_keywords = client.fetch_keyword_trends(keyword_config)
        print("\n키워드 트렌드 분석 결과:")
        print(df_keywords)

        # 키워드 디바이스별 분석
        keyword_device_config = KeywordRequestConfig(
            start_date="2024-01-01",
            end_date="2024-04-01",
            category="50000006",
            keyword="건강",
            ages=["20", "30", "40"]
        )
        df_keyword_device = client.fetch_keyword_device_traffic(
            keyword_device_config)
        print("\n키워드 디바이스별 분석 결과:")
        print(df_keyword_device)

        # 키워드 성별 분석
        keyword_gender_config = KeywordRequestConfig(
            start_date="2024-01-01",
            end_date="2024-04-01",
            category="50000006",
            keyword="건강",
            ages=["20", "30", "40"]
        )
        df_keyword_gender = client.fetch_keyword_gender_traffic(
            keyword_gender_config)
        print("\n키워드 성별 분석 결과:")
        print(df_keyword_gender)

        # 키워드 연령대별 분석
        keyword_age_config = KeywordRequestConfig(
            start_date="2022-01-01",
            end_date="2024-04-01",
            time_unit=TimeUnit.DATE,
            category="50000006",
            keyword="건강",
            ages=["20", "30", "40", "50", "60"]
        )
        df_keyword_age = client.fetch_keyword_age_traffic(keyword_age_config)
        print("\n키워드 연령대별 분석 결과:")
        print(df_keyword_age)

    except Exception as e:
        print(f"에러 발생: {str(e)}")


if __name__ == "__main__":
    main()
