from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain import VectorDBQA
from langchain.chains import RetrievalQA
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI as OpenAI
import yaml

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    if 'HTTPS_PROXY' in config:
        if os.environ.get('HTTPS_PROXY') is None:   # 优先使用环境变量中的代理，若环境变量中没有代理，则使用配置文件中的代理
            os.environ['HTTPS_PROXY'] = config['HTTPS_PROXY']
    PORT = config['PORT']
    CHAT_CONTEXT_NUMBER_MAX = config['CHAT_CONTEXT_NUMBER_MAX']     # 连续对话模式下的上下文最大数量 n，即开启连续对话模式后，将上传本条消息以及之前你和GPT对话的n-1条消息
    USER_SAVE_MAX = config['USER_SAVE_MAX']   
    
from langchain.prompts import PromptTemplate
prompt_template = """使用以下 文本 来回答最后的 问题。
如果你不知道答案，只回答"未找到答案"，不要编造答案。
如果你的答案不是来自 文本 ，只回答"未找到答案"，不要根据你已有的知识回答。
答案应该尽量流畅自然，答案应该尽量完整。
你必须使用中文回答。

文本: {context}

问题: {question}
中文答案:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# test the role of metadata in the retrieve process
def test_metadata_retrieve():
    API_KEY = os.getenv("OPENAI_API_KEY")
    docs=[]
    # 测试文本
    # docs.append(
    #     Document(page_content="今天中午吃牛排",metadata={})
    # )
    # 测试metadata key+value
    docs.append(
        Document(page_content="",metadata={'午餐':"鸡腿"})
    )
    docs.append(
        Document(page_content="",metadata={'午餐':"卤肉饭"})
    )
    docs.append(
        Document(page_content="",metadata={'午餐':"西红柿炒鸡蛋"})
    )
    docs.append(
        Document(page_content="",metadata={'午餐':"汤圆"})
    )
    docs.append(
        Document(page_content="",metadata={'午餐':"饺子"})
    )
    # 不相关输入
    docs.append(
        Document(page_content="",metadata={'天气':"晴"})
    )
    docs.append(
        Document(page_content="",metadata={'天气':"多云"})
    )
    docs.append(
        Document(page_content="",metadata={'天气':"阵雨"})
    )
    docs.append(
        Document(page_content="",metadata={'午餐':"鸡排"})
    )
    docs.append(
        Document(page_content="大海是蓝色的",metadata={})
    )
    # 测试metadata value
    docs.append(
        Document(page_content="",metadata={'meal':"今天中午吃奶油蘑菇汤"})
    )
    docs.append(
        Document(page_content="",metadata={'meal':"烤生蚝"})
    )
    embeddings = OpenAIEmbeddings()
    docsearch=Chroma.from_documents(docs,embeddings)
    chain_type_kwargs = {"prompt": PROMPT}
    chain = RetrievalQA.from_chain_type(llm=OpenAI(model_name="gpt-3.5-turbo",max_tokens=500,temperature=0), chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs,verbose=True,return_source_documents=True)
    print(chain({"query":"今天中午吃什么？"}))

test_metadata_retrieve()