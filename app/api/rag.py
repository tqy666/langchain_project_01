# -*- coding: utf-8 -*-
from fastapi import APIRouter, File, UploadFile,Depends
from pathlib import Path
from app.ragServer.ragUploade import RagUploader
from starlette.responses import StreamingResponse
from app.ragServer.ragServerTool import RagServerTool
from schemas.chat import ChatRequest


router = APIRouter()
_uploader = RagUploader()
_RagServer = RagServerTool()


@router.post("/rag/uplode")
async def uplode_files(files: UploadFile = File(...)):
    """
    上传文件接口。
    """
    uploader = _uploader


    # 1.检查文件的类型
    suffix_data = {".pdf", ".docx", ".pptx", ".xlsx"}
    print(f"文件类型是：{Path(files.filename).suffix}")
    if Path(files.filename).suffix not in suffix_data:
        return {"msg": "文件类型不支持"}

    #2.检查文件的大小<10 MB
    if files.size >=10 * 1024 * 1024:
        return {"msg":"文件上传最大不能超过10M"}

    #3. 检查上传的内容是否在重复
    await files.seek(0)

    file_md5 = await uploader.check_content_file(files)
    print(f"md5加密后的文件查询有没有是：{file_md5}")
    if file_md5:
        return {"msg":"该文件已经上传过了，请换一个新文件上传"}
    await files.seek(0)



    #4.处理单个文件的 RAG 流程
    #注意：这里直接传入 Path 对象，由 RagUploader 内部处理转换
    param = await uploader.get_upload_file(files)
    if "error" in param:
        msg = param.get("detail") or param.get("message") or param.get("error", "文件处理失败")
        return {"msg": msg, "data": param, "code": 400}

    if param.get("chunks", 1) == 0 and "error" not in param:
        return {"msg": param.get("message", "文件处理失败"), "data": param, "code": 400}

    return {"msg": "上传成功", "data": param, "code": 200}

@router.get("/rag/getFileList")
async def getFileList():
    """
           获取文件列表
           """
    return _uploader.getFileList()
@router.delete("/rag/del/{file_id}")
async def del_files(file_id: str):
    """
        删除单个文件
        """
    param = _uploader.del_file(file_id)

    return param


@router.post("/rag/stream")
async def helpdesk(request: ChatRequest):
    """
        流式对话接口。
        根据用户的提问，通过稠密检索,BM25,多路召回，重排序，返回用户所需要的知识
        """
    return StreamingResponse(
        _RagServer.getKnowledge(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@router.get("/rag/history_list")
async def read_chat_history_fixed():
    """
     默认是：threat_id:project_01 前端不用传值，项目为测试项目，后端固定了一个测试
     获取的是聊天列表
    """

    data =  _RagServer.read_chat_history_fixed()
    return data

@router.get("/rag/history/{thread_id}")
async def get_chat_info(thread_id:str):
    """
    前端传checkpoint_id后可以查询对应聊天记录
    :param checkpoint_id:
    :return:
    """
    chat_info = _RagServer.get_chat_info(thread_id)
    return chat_info