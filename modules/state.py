import streamlit as st
import os

DEFAULT_STATE = {
    "api_key": "",
    "temperature": 0.7,
    "max_tokens": 512,
    "top_p": 1,
    "model": "gpt-3.5-turbo",
}

TYPES_STATE = {
    "temperature": float,
    "max_tokens": int,
    "top_p": float
}

def read_environ_params():
    env = os.environ
    prefix = "SECRET_"
    return {k.replace(prefix, "").lower(): env[k]  for k in env.keys() if k.startswith(prefix)}


def read_url_param_values():
    urlparams = st.experimental_get_query_params()
    return {
        **{k: TYPES_STATE.get(k, str)(
            urlparams.get(k, [v])[0]) 
        for k, v in DEFAULT_STATE.items()}, 
        **read_environ_params()}


def set_url_param_value(key):
    current = read_url_param_values()
    st.experimental_set_query_params(
        **{
            **current,
            **{key: st.session_state.get(key, current[key])}}
    )