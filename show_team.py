import streamlit as st
from retention_main import GraphPloter, FirebaseInfoFetcher

st.title('토리숲 Growth Analysis 🐿️ 📊')
st.write('')

# 세션 상태 초기화
if 'show_retention' not in st.session_state:
    st.session_state['show_retention'] = False
if 'show_activation' not in st.session_state:
    st.session_state['show_activation'] = False
if 'show_others' not in st.session_state:
    st.session_state['show_others'] = False

# 다섯 개의 동일한 비율의 열 생성
col1, col2, col3, col4, col5, col6, col7 = st.columns([1,3,1,3,1,3,1])

# 두 번째 열에 'Retention' 버튼 배치
with col2:
    if st.button('Retention'):
        st.session_state['show_retention'] = True
        st.session_state['show_activation'] = False  
        st.session_state['show_others'] = False


# 네 번째 열에 'Activation' 버튼 배치
with col4:
    if st.button('Activation'):
        st.session_state['show_activation'] = True
        st.session_state['show_retention'] = False  
        st.session_state['show_others'] = False


with col6:
    if st.button('Others'):
        st.session_state['show_activation'] = False
        st.session_state['show_retention'] = False  
        st.session_state['show_others'] = True




if st.session_state['show_others']:
    fetcher = FirebaseInfoFetcher(other=True)
    st.write('Hello')
    st.write(f'Total Download User: {fetcher.total_downloaded_user_num}')
    st.write(f'Total Activated User: {fetcher.activated_user_num}')
    st.write(f'Get Toal Conversation Number:{fetcher.chat_num}')


# Retention 콘텐츠 표시 (전체 너비)
if st.session_state['show_retention']:
    graph_ploter = GraphPloter(bracket=1) ##bracket도 클라이언트 요청으로 동적으로 받음
    st.write('')

    with st.container(border=True):
        st.write(':orange[코호트 데이터프레임]')
        st.write(graph_ploter.df)
        st.write('')

    with st.container(border=True):
        st.write(':orange[코호트 리텐션 히트맵]')
        st.pyplot(graph_ploter.heatmap)
        st.write('')

    with st.container(border=True):
        st.write(':orange[리텐션 그래프]')
        st.pyplot(graph_ploter.retention_graph)


# Activation 콘텐츠 표시 (전체 너비)
if st.session_state['show_activation']:
    graph_ploter = GraphPloter(bracket=1, funnel=True) ##bracket도 클라이언트 요청으로 동적으로 받음

    st.write('')

    with st.container(border=True):
        st.write(':orange[activation 통과율]')
        st.pyplot(graph_ploter.activation_linegraph)


