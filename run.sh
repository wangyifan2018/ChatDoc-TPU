#!/bin/bash
set -ex

res=$(which unzip)

if [ $? != 0 ];
then
    echo "Please install unzip on your system!"
    exit
fi

pip install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade

# default param
llm_model="chatglm3"
dev_id="0"
# Args
parse_args() {
    while [[ $# -gt 0 ]]; do
        key="$1"

        case $key in
            --model)
                llm_model="$2"
                shift 2
                ;;
            --dev_id)
                dev_id="$2"
                shift 2
                ;;
            *)
                echo "Invalid option: $key" >&2
                exit 1
                ;;
            :)
                echo "Option -$OPTARG requires an argument." >&2
                exit 1
                ;;
        esac
    done
}

# Process Args
parse_args "$@"


# nltk_data & bert_model is required
if [ ! -d "$HOME/nltk_data" ]; then
    echo "$HOME/nltk_dat does not exist, download..."
    python3 -m dfss --url=open@sophgo.com:ezoo/chatdoc/nltk_data.zip
    unzip nltk_data.zip
    mv nltk_data ~
    rm nltk_data.zip
    echo "nltk_data download!"
else
    echo "$HOME/nltk_dat already exist..."
fi

# download bert_model
if [ ! -d "./models/bert_model" ]; then
    echo "./models/bert_model does not exist, download..."
    python3 -m dfss --url=open@sophgo.com:ezoo/chatdoc/bert_model.zip
    unzip bert_model.zip -d ./models
    rm bert_model.zip
    echo "bert_model download!"
else
    echo "$HOME/nltk_dat already exist..."
fi

# download LLM models
if [ "$llm_model" == "chatglm3" ]; then
    if [ ! -d "./models/glm3_model" ]; then
        echo "./models/glm3_model does not exist, download..."
        python3 -m dfss --url=open@sophgo.com:ezoo/chatdoc/glm3_model.zip
        unzip glm3_model.zip -d ./models
        rm glm3_model.zip
        echo "glm3_model download!"
    else
        echo "./models/glm3_model already exist..."
    fi
elif [ "$llm_model" == "qwen" ]; then
    if [ ! -d "./models/qwen_model" ]; then
        echo "./models/qwen_model does not exist, download...."
        python3 -m dfss --url=open@sophgo.com:ezoo/chatdoc/qwen_model.zip
        unzip qwen_model.zip -d ./models
        rm qwen_model.zip
        echo "qwen_model download!"
    else
        echo "./models/qwen_model already exist..."
    fi
else
    echo "Error: --model is not recognized. Must be 'chatglm3' or 'qwen'."
    exit 1
fi


export LLM_MODEL=$llm_model
export DEVICE_ID=$dev_id

streamlit run web_demo_st.py --server.address '0.0.0.0'
