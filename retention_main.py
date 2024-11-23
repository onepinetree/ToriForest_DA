import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

class FirebaseInfoFetcher:

    def __init__(self, other = False, funnel = False):
        if not firebase_admin._apps:
            cred = credentials.Certificate("etc/secrets/dotori-fd1b0-firebase-adminsdk-zzxxd-fb0e07e05e.json")
            firebase_admin.initialize_app(cred)
        self.db = firestore.client()
        if other == False:
            self.user_retention_data = self.getUserRetentionData()
            if funnel == True:
                self.user_funnel_data = self.getuserFunnelData()
        else:
            self.total_downloaded_user_num : int = self.getTotalDownloadNum()
            self.activated_user_num :int = self.getActivatedUserNum()
            self.chat_num = self.getAllChatNum()


    def getUserRetentionData(self) -> dict:
        '''retentionData에 들어있는 user_data의 형식에 맞게 data를 return 하는 함수'''

        retention_doc_ref = self.db.collection('staffOnly').document('retentionData')

        user_data = {}
        for user_uid, active_date_dict in retention_doc_ref.get().to_dict().items():
            if user_uid == 'dummy':
                continue
            user_active_date = []
            for date_key in active_date_dict:
                user_active_date.append(date_key)

            user_data[user_uid] = self.sort_dates(date_list=user_active_date)
        
        return user_data

    @staticmethod
    def sort_dates(date_list) -> list:
        # 문자열 날짜 리스트를 datetime 객체 리스트로 변환 후 정렬
        sorted_dates = sorted(date_list, key=lambda date: datetime.strptime(date, '%Y-%m-%d'))
        return sorted_dates
    
    def getuserFunnelData(self) -> dict:
        '''유저들의 콜렉션 Id와 최종 단계를 key-value로 한 dict을 return 하는 함수'''
        excluded_collections = ['staffOnly', 'userInfo', '문의하기']
        results = {}
        
        # Retrieve all collections
        collections = self.db.collections()
        for collection in collections:
            collection_name = collection.id
            if collection_name in excluded_collections:
                continue  # Skip excluded collections
                
            # Access the 'info' document in the current collection
            info_doc_ref = collection.document('info')
            info_doc = info_doc_ref.get()
            
            if info_doc.exists:
                fields = info_doc.to_dict()

                name_value = fields.get('name', '')
                nickname_value = fields.get('nickname', '')

                if name_value == '익명':
                    continue
                has_nickname = bool(str(nickname_value).strip())
                
                if not has_nickname:
                    result = 1  # 닉네임이 존재하지 않음 (1단계만 통과)
                else:

                    chat_doc_ref = collection.document('chat')
                    notes_doc_ref = collection.document('notes')
                    
                    chat_doc = chat_doc_ref.get()
                    notes_doc = notes_doc_ref.get()
                    
                    chat_exists = chat_doc.exists
                    notes_exists = notes_doc.exists

                    if not chat_exists:
                        result = 2 # chat document가 아예 존재하지 않음 (2단계만 통과)
                    else:
                        chat_data = chat_doc.to_dict()

                        result = 3 #채팅방에는 들어왔는데 아무런 채팅은 하지 않음 (3단계 통과)
                        
                        for date, chat_info in chat_data.items():
                            length_of_chat = len(chat_info) - 2
                            if length_of_chat > 2:
                                if length_of_chat > 6:
                                    if notes_exists:
                                        result = 6 #모든 퍼널 완료한 사람
                                    else:
                                        result = 5 #대화는 했지만 정리는 안한 사람
                                else:
                                    result = 4 #채팅방에서 3번이상 답변을 하지는 않은 사람
            else:
                result = 1  # 'info' document does not exist
                
            # Store the result for the current collection
            results[collection_name] = result
            
        # Print the results in the specified format
        # for collection_name, status in results.items():
        #     print(f"Collection '{collection_name}': {status}")
            
        return results
    

    def getTotalDownloadNum(self)-> int:
        return len(list(self.db.collections())) - 3

    def getActivatedUserNum(self)->int:
        retention_doc_ref = self.db.collection('staffOnly').document('retentionData')
        return len(retention_doc_ref.get().to_dict())

    def getAllChatNum(self) -> int:
        num = 0
        for collection in self.db.collections():
            chat_doc = collection.document('chat').get()
            if chat_doc.exists:
                num += len(chat_doc.to_dict())
        return num


class RetentionAnalyser:
    '''Retention DataFrame을 return 하는 함수'''

    def __init__ (self, bracket):
        self.user_data = FirebaseInfoFetcher().user_retention_data
        self.all_dates = self.generateDates()
        self.start_date = min(self.all_dates).strftime('%Y-%m-%d')
        self.end_date = max(self.all_dates).strftime('%Y-%m-%d')
        self.cohort_data = self.buildCohortData(start_date=self.start_date, end_date=self.end_date, bracket=bracket)
        self.df = self.dict_to_dataframe(cohort_data=self.cohort_data)

    def generateDates(self):
    # 사용자 데이터에서 시작 날짜와 종료 날짜를 동적으로 설정
        date_format = '%Y-%m-%d'
        all_dates = []
        for dates in self.user_data.values():
            date_objs = [datetime.strptime(date_str, date_format) for date_str in dates]
            all_dates.extend(date_objs)
        
        return all_dates


    @staticmethod
    def dates_to_binary_list(date_list: list):
        """주어진 날짜 리스트를 이진 리스트로 변환합니다."""
        date_format = '%Y-%m-%d'
        try:
            # 문자열을 datetime 객체로 변환
            date_list = [datetime.strptime(date_str, date_format) for date_str in date_list]
        except ValueError as e:
            print(f"날짜 형식이 잘못되었습니다: {e}")
            return []
        date_set = set(date_list)
        start_date = min(date_list)
        end_date = max(date_list)
        delta_days = (end_date - start_date).days + 1

        binary_list = []
        for i in range(delta_days):
            current_date = start_date + timedelta(days=i)
            if current_date in date_set:
                binary_list.append(1)
            else:
                binary_list.append(0)
        return binary_list

    def modified_dates_to_binary_list(self, date_list: list):
        """날짜 리스트에 오늘 날짜를 추가하고 마지막 1을 제거하여 이진 리스트로 변환합니다."""
        date_format = '%Y-%m-%d'
        today = datetime.now().strftime(date_format)
        try:
            # 문자열을 datetime 객체로 변환
            date_list_dt = [datetime.strptime(date_str, date_format) for date_str in date_list]
            today_dt = datetime.strptime(today, date_format)
        except ValueError as e:
            print(f"날짜 형식이 잘못되었습니다: {e}")
            return []

        # 마지막 날짜가 오늘 날짜가 아닌 경우
        if date_list_dt[-1] != today_dt:
            date_list.append(today)
            binary_list = self.dates_to_binary_list(date_list)
            # 마지막 1 제거
            for i in range(len(binary_list)-1, -1, -1):
                if binary_list[i] == 1:
                    binary_list[i] = 0
                    break
            # binary_list.pop()
            return binary_list
        else:
            # 오늘 날짜가 이미 포함된 경우 그대로 실행
            # self.dates_to_binary_list(date_list).pop()
            return self.dates_to_binary_list(date_list)

    #사용 예시
    # date_list = ['2024-11-17', '2024-11-18']
    # print(dates_to_binary_list(date_list))
    # print(modified_dates_to_binary_list(date_list))

    @staticmethod
    def modifyIntoBracket(lst : list, bracket : int) -> list:
        '''양의 정수의 bracket을 넣으면 binary_list를 bracket_retention 측정에 맞게 변형해주는 함수'''
        return [
            1 if any(lst[i:i + bracket]) else 0 
            for i in range(0, len(lst) - len(lst) % bracket, bracket)
        ]

    @staticmethod
    def combine_binary_lists_to_average_list(binary_lists):
        '''리텐션에 대한 바이너리 리스트들을 담고 있는 리스트의 평균을 구합니다'''
    
        max_length = max(len(lst) for lst in binary_lists)
        padded_lists = [lst + [np.nan]*(max_length - len(lst)) for lst in binary_lists]
        df = pd.DataFrame(padded_lists)
        
        # 각 컬럼의 평균을 계산하여 백분율로 변환합니다.
        averages = df.mean(skipna=True) * 100
        
        # 평균 값의 소수점을 조절하고 백분율 기호를 추가합니다.
        averages = averages.round(2).astype(str) + '%'
        
        # 리스트로 변환하여 반환합니다.
        return averages.tolist()

    # binary_lists = [
    #     [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    #     [1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    #     [1, 0, 0, 0, 0, 0, 0, 0, 1]
    # ]

    # print(combine_binary_lists_to_average_list(binary_lists))

    @staticmethod
    def create_date_list(start_date, end_date):
        '''시작 날짜와 끝 날짜를 넣으면 사이 날짜를 채운 리스트를 return 하는 함수'''
        date_list = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current_date <= end_date_obj:
            date_list.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
        
        return date_list

    # print(create_date_list('2024-12-30', '2025-1-14'))

    @staticmethod
    def dict_to_dataframe(cohort_data):
        # 각 리스트의 최대 길이를 구합니다.
        max_length = max(len(info['cohort_retention_rate']) for info in cohort_data.values())
        
        # 열 이름을 'cohort_num', 'Day_0', 'Day_1', ... 형식으로 만듭니다.
        column_names = ['cohort_num'] + [f'Day_{i}' for i in range(max_length)]
        
        # 데이터를 변환하여 데이터프레임을 생성합니다.
        data = {}
        for key, value in cohort_data.items():
            # cohort_num 저장
            cohort_num = value['cohort_num']
            
            # 리스트가 짧은 경우 NaN으로 채워줍니다.
            padded_list = value['cohort_retention_rate'] + [np.nan] * (max_length - len(value['cohort_retention_rate']))
            
            # 첫 번째 열로 cohort_num을 넣어줍니다.
            data[key] = [cohort_num] + padded_list
        
        # 데이터프레임 생성
        df = pd.DataFrame.from_dict(data, orient='index', columns=column_names)
        
        # 'Average' 행 계산 및 추가
        retention_columns = df.columns[1:]  # 'cohort_num' 제외한 나머지 열
        total_cohort_num = df['cohort_num'].sum()
        print(total_cohort_num)
        weighted_averages = []
        
        for col in retention_columns:
            valid_rows = df[col].notna()
            if valid_rows.any():
                weights = df.loc[valid_rows, 'cohort_num']
                # 퍼센트 문자열을 실수로 변환
                values = df.loc[valid_rows, col].str.rstrip('%').astype(float)
                weighted_avg = np.sum(values * weights) / np.sum(weights)
                weighted_averages.append(f'{weighted_avg:.2f}%')
            else:
                weighted_averages.append(np.nan)
        
        df.loc['Average'] = [total_cohort_num] + weighted_averages
        
        # 인덱스 재정렬: 'Average'를 가장 위로, 나머지는 날짜 순으로 정렬
        date_indices = [idx for idx in df.index if idx != 'Average']
        # 날짜 문자열을 datetime 객체로 변환하여 정렬
        date_indices_sorted = sorted(date_indices, key=lambda x: datetime.strptime(x, '%Y-%m-%d'))
        # 새로운 인덱스 순서로 재정렬
        df = df.reindex(['Average'] + date_indices_sorted)
        
        return df



    def buildCohortData(self, start_date: str, end_date: str, bracket: int):
        '''코호트 데이터를 생성해서 return 하는 함수'''
        #코호트 별로 평균과 코호트 구성원 수를 가지고 있는 dict
        cohort_data = {}
        for date in self.create_date_list(start_date=start_date, end_date=end_date):
            #같은 코호트끼리의 핵심이벤트 수행 리스트를 모은 리스트
            cohortUserKeyEventDates = []
            for user, userKeyEventDates in self.user_data.items(): 
                if userKeyEventDates[0] == date:
                    binary_list = self.modified_dates_to_binary_list(userKeyEventDates)
                    binary_list = self.modifyIntoBracket(lst=binary_list, bracket=bracket)
                    if binary_list:                                                                                                                                                                                                                                                                                                                                                                                                                                
                        cohortUserKeyEventDates.append(binary_list)
            if len(cohortUserKeyEventDates) > 0:
                #같은 길이의 list들을 average하게 됨
                averages = self.combine_binary_lists_to_average_list(cohortUserKeyEventDates)
                # 백분율 형식으로 변환
                averages = [f"{avg}%" for avg in averages]
                averages.pop()
                cohort_data[date] = {
                    'cohort_num': len(cohortUserKeyEventDates),
                    'cohort_retention_rate': averages
                }
        return cohort_data

from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


import platform
import matplotlib as mpl


class GraphPloter:

    def __init__(self, bracket, funnel = False):
        if platform.system() == 'Darwin':
            mpl.rcParams['axes.unicode_minus'] = False

        if funnel == True:
            self.user_funnel_data = FirebaseInfoFetcher(funnel = True).user_funnel_data
            self.total_users_num = len(self.user_funnel_data)
            self.activation_linegraph = self.visualize_activation_funnel()
        else:
            self.df = RetentionAnalyser(bracket=bracket).df
            self.retention_graph = self.plot_average_retention_with_plateau()
            self.heatmap = self.plot_heatmap()



    def plot_average_retention_with_plateau(self):
        # 'Average' 행 추출
        average_row = self.df.loc['Average']

        # 'cohort_num' 가져오기
        total_cohort_num = average_row['cohort_num']

        # 'cohort_num'을 제외한 유지율 데이터 추출
        retention_rates = average_row.drop('cohort_num')

        # '%' 기호 제거 및 실수형 변환
        retention_rates = retention_rates.str.rstrip('%').astype(float)

        # x축 라벨 생성
        x_labels = retention_rates.index.tolist()
        x_values = np.arange(len(x_labels))

        # 유지율 변화 계산
        changes = retention_rates.diff().abs()
        # 변화율이 threshold 이하인 지점 찾기
        threshold = 0.5  # 0.5% 이하의 변화
        plateau_start = None
        for i in range(1, len(changes)):
            if changes.iloc[i] <= threshold:
                plateau_start = i
                break

        # 그래프 객체 생성
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(x_values, retention_rates, color='black', marker='o')

        # Plateau 영역 강조
        if plateau_start is not None:
            ax.axvspan(x_values[plateau_start], x_values[-1], color='lightgrey', alpha=0.5, label='Retention Plateau')

        # 배경색 설정
        ax.set_facecolor('skyblue')

        # 각 포인트에 퍼센트 값 표시
        for i, value in enumerate(retention_rates):
            ax.text(x_values[i], retention_rates.iloc[i] + 1, f'{value:.2f}%', ha='center', va='bottom', color='black')

        # x축 라벨 설정
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels)

        # y축 라벨 설정
        ax.set_ylabel('Retention Rate (%)')

        # 제목 설정
        ax.set_title(f'Retention Curve of {int(total_cohort_num)} Users')

        # 범례 추가
        if plateau_start is not None:
            ax.legend()

        # 그리드 추가
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.close(fig)  # 그래프를 닫아 메모리 누수 방지
        return fig  # Figure 객체 반환


    def plot_heatmap(self):
        # 'cohort_num' 열 제외하고 복사본 생성
        df_numeric = self.df.drop(columns=['cohort_num']).copy()
        
        # 퍼센트 문자열을 숫자(float)로 변환
        for col in df_numeric.columns:
            df_numeric[col] = df_numeric[col].astype(str).map(
                lambda x: float(x.rstrip('%')) if '%' in x else np.nan
            )
        
        # 히트맵에서 'Average'가 가장 위에 오고, 나머지는 날짜의 빠른 순으로 정렬되도록 인덱스 재정렬
        date_indices = [idx for idx in df_numeric.index if idx != 'Average']
        date_indices_sorted = sorted(date_indices, key=lambda x: datetime.strptime(x, '%Y-%m-%d'))
        new_index_order = ['Average'] + date_indices_sorted
        df_numeric = df_numeric.loc[new_index_order]
        
        # 그래프 객체 생성
        nrows, ncols = df_numeric.shape
        cell_size = 0.8
        fig, ax = plt.subplots(figsize=(ncols * cell_size, nrows * cell_size))
        
        # 값을 퍼센트 형식으로 표시
        annot_data = df_numeric.applymap(lambda x: f'{x:.1f}%' if not pd.isna(x) else '')
        
        sns.heatmap(
            df_numeric, 
            cmap='Blues', 
            annot=annot_data,
            fmt="", 
            linewidths=.5, 
            annot_kws={'size': 7.6},  # 여기서 글자 크기를 설정 (예: 8)
            cbar_kws={'orientation':'vertical'},
            ax=ax  # ax 인자 추가
        )
        
        # 컬럼 이름을 히트맵 상단에 표시
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        
        # 레이블 설정
        ax.set_title('Retention Rate Heatmap')
        ax.set_ylabel('Cohort')

        # 행(row)와 열(column) 레이블의 크기 조정
        ax.tick_params(axis='x', labelsize=9)  # 열 레이블 크기 조정
        ax.tick_params(axis='y', labelsize=9)  # 행 레이블 크기 조정
        
        plt.tight_layout()
        plt.close(fig)  # 그래프를 닫아 메모리 누수 방지
        return fig  # Figure 객체 반환
    

    def visualize_activation_funnel(self):

        total_users = self.total_users_num
        # 단계별 사용자 수 초기화
        stage_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        
        # 각 단계에서의 사용자 수 계산
        for status in self.user_funnel_data.values():
            for stage in range(1, status + 1):
                if stage in stage_counts:
                    stage_counts[stage] += 1
        
        # 각 단계에서의 사용자 비율 계산
        stages = sorted(stage_counts.keys())
        percentages = [(stage_counts[stage] / total_users) * 100 for stage in stages]
        
        # 각 활성화 단계에 대한 커스텀 라벨 정의
        stage_labels = [
            'Email, Password, \nName input Screen',
            'First Meeting Tori',
            'Enter chat',
            'Chat almost once\n(less than three)',
            'Chat but not\n Dotori Saved',
            'First Dotori \nSaved'
        ]
        
        # 활성화 퍼널 그래프 생성
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(stages, percentages, color='skyblue')
        ax.set_title(f'Activation Funnel Through Percentage of {total_users} Users', fontsize=20)
        ax.set_xlabel('Activation Funnel', fontsize=12)
        ax.set_ylabel('User Ratio (%)', fontsize=12)
        ax.set_xticks(stages)
        ax.set_xticklabels(stage_labels, rotation=0, ha='center', fontsize=10)
        ax.set_ylim(0, 100)
        ax.grid(axis='y')
        plt.tight_layout()  # 레이블이 잘리지 않도록 레이아웃 조정
        plt.close(fig)  # 메모리 누수 방지를 위해 그래프 닫기
        return fig  # Figure 객체 반환

        # 각 단계의 비율 출력 (선택 사항)
        # for stage, percentage, label in zip(stages, percentages, stage_labels):
        #     print(f"Stage {stage} ({label}): {percentage:.2f}% users passed")



