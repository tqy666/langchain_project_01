# -*- coding: utf-8 -*-
from langchain_chroma import Chroma
from langchain_mineru import MinerULoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from pathlib import Path
from fastapi import  UploadFile
import os
import hashlib
from config.MySqlconfig import MysqlConfig
from uuid import uuid4
import aiofiles
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime

class RagUploader:
    def __init__(self):
        # self.embeddings = OllamaEmbeddings(
        #     model="qwen3-embedding:0.6b",
        #     dimensions=1024
        # )
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")  #电脑内存太小了，1024分配不到内存，默认用nomic768
        # 预定义存储路径，避免重复计算
        self.persist_dir = Path(__file__).resolve().parent.parent.parent / "chroma_dir"
        self.persist_dir.mkdir(exist_ok=True)  # 确保目录存在
        self.mysqldb = MysqlConfig()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.collection_name = "nomic_embeddings_v3"
        self.vector_db = Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=self.embeddings,
            collection_name=self.collection_name
        )
        self.hash_mad5 = None


    async def _save_upload_file(
        self,
        upload_file: UploadFile,
        dest: Path,
        chunk_size: int = 4 * 1024 * 1024,
    ) -> int:
        """流式写入上传文件，避免整文件读入内存。"""
        total = 0
        async with aiofiles.open(dest, "wb") as f:
            while chunk := await upload_file.read(chunk_size):
                await f.write(chunk)
                total += len(chunk)
            await f.flush()
        return total

    def search_sql(self,field_name,field_val):
        sql = f"SELECT * FROM file_chunk WHERE {field_name} = %s LIMIT 1"

        result = self.mysqldb.query(sql, (field_val,))
        return result

    async def check_content_file(self,file: UploadFile, chunk_size: int = 1024 * 1024):
        """异步分块计算 MD5"""
        md5_hash = hashlib.md5()

        while chunk := await file.read(chunk_size):
            md5_hash.update(chunk)

        md5_data = md5_hash.hexdigest()
        self.hash_mad5 = md5_data

        return  self.search_sql("md5_hash",md5_data)

    async def _safe_upload_path(self,filename: str) -> Path:
        """生成安全的上传路径，防止路径穿越攻击"""

        suffix = Path(filename).suffix

        safe_name = f"{uuid4().hex}{suffix}"
        upload_dir = Path(__file__).resolve().parent.parent.parent / "uploads"
        upload_dir.mkdir(exist_ok=True)  # 确保上传目录存在
        return upload_dir / safe_name

    async def get_upload_file(self, files: UploadFile) -> Dict[str, Any]:
        """
        异步处理文件上传、解析、切分和入库。
        文件落盘采用流式写入；MinerU 解析与向量入库在线程池执行，避免阻塞事件循环。
        """
        loop = asyncio.get_running_loop()
        file_paths = await self._safe_upload_path(files.filename)

        try:
            await files.seek(0)
            file_size = await self._save_upload_file(files, file_paths)
            if file_size == 0:
                return {
                    "message": "文件读取失败，内容为空，请检查上传的文件是否有效。",
                    "error": "empty_file",
                    "chunks": 0,
                }

            print(f"文件写入成功: {file_paths}, 大小: {file_size} 字节")

            def _load_docs():
                try:
                    loader = MinerULoader(source=[str(file_paths)], mode="flash")
                    return loader.load()
                except Exception as e:
                    print(f"MinerU 内部加载错误: {e}")
                    return None

            raw_docs = await loop.run_in_executor(self.executor, _load_docs)

            if not raw_docs or not isinstance(raw_docs, list) or len(raw_docs) == 0:
                # 如果 MinerU 返回空列表或 None，尝试检查文件是否存在且非零
                if file_size == 0:
                    return {"message": "文件写入失败", "error": "文件大小为0字节"}
                else:
                    return {
                        "message": "文档解析失败",
                        "error": "MinerU 未能提取内容",
                        "detail": "文件格式可能不受支持或已损坏，请检查是否为标准 Word/PDF 文件。"
                    }

            print(f"成功解析出 {len(raw_docs)} 个文档片段")

            # 3. 文档切分 (这部分通常很快，也可以放在线程里，或者保持同步)
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
            )
            chunks = text_splitter.split_documents(raw_docs)
            len_chunks= len(chunks)
            print(f"成功切分出 {len_chunks} 个文本块")

            filename = os.path.basename(file_paths)
            ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
            print(f"ids打印：{ids}")

            # 4. 存进向量库
            def _add_to_vector_db():
                self.vector_db.add_documents(documents=chunks, ids=ids)

            await loop.run_in_executor(self.executor, _add_to_vector_db)


            # if self.db is not None:
            #     self.db.add_documents(chunks, ids=ids,collection_name=self.collection_name)
            # else:
            #     self.db = Chroma.from_documents(
            #         chunks,
            #         self.embeddings,
            #         persist_directory=str(self.persist_dir),
            #         ids=ids,
            #         collection_name=self.collection_name
            #     )

            # 5. 向量库和原始文件相关的数据存入 MySQL，供后续删除使用

            insert_sql = """
                INSERT INTO file_chunk (
                    md5_hash, collections, filename, file_size,
                    chunk, file_path, ids, creat_time
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s )
            """
            insert_params = (
                self.hash_mad5,
                self.collection_name,
                files.filename,
                f"{file_size / (1024 * 1024):.2f}M",
                len_chunks,
                str(file_paths),
                json.dumps(ids, ensure_ascii=False),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )

            print(f"写入 file_chunk: filename={files.filename}, chunks={len_chunks}")
            if not self.hash_mad5:
                return {
                    "message": "文件处理失败",
                    "error": "missing_md5",
                    "detail": "未获取文件 MD5，请先完成重复校验后再上传。",
                }

            result = self.mysqldb.insert(insert_sql, insert_params)
            if result:
                return {
                    "message": "知识库更新,插入数据完成。",
                    "chunks": len(chunks),
                    "persist_directory": str(self.persist_dir)
                }
            else:
                return {
                    "message": "知识库更新,插入数据失败。",
                    "chunks": len(chunks),
                    "persist_directory": str(self.persist_dir)
                }


        except ValueError as ve:
            # 捕获 MinerU 特定的解析失败错误
            # 日志记录详细的错误信息
            print(f"MinerU 解析失败: {ve}")
            # 返回友好的错误信息，而不是让接口崩溃
            return {
                "message": "文件解析失败",
                "error": str(ve),
                "detail": "MinerU 服务未能成功解析该文档，请检查文件是否损坏或稍后重试。"
            }
        except Exception as e:
            print(f"处理文件时发生未知错误: {e}")
            return {
                "message": "文件处理失败",
                "error": str(e),
                "detail": "上传或入库过程中发生异常，请稍后重试。",
            }

    def getFileList(self):
        sql = f"SELECT * FROM file_chunk order by creat_time desc"

        result = self.mysqldb.query(sql)
        return result

    def del_file(self,file_id,field_name="id"):

        seach_file = self.search_sql(field_name,file_id)
        ids = json.loads(seach_file[0]['ids'])
        print(f"删除前总共的条数{self.vector_db._collection.count()}")
        #删除向量库里的数据
        self.vector_db.delete(ids=ids)
        print(f"删除后总共的条数{self.vector_db._collection.count()}")

        #删除本地文件f
        file_path = seach_file[0]['file_path']
        print(f"要删除的本地文件：{file_path}")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"物理文件 {file_path} 已成功删除")
            except Exception as e:
                print(f"删除物理文件失败: {e}")
        else:
            print("该文件在磁盘上不存在")
        sql = f"DELETE FROM file_chunk WHERE id = %s "
        result = self.mysqldb.execute(sql, (file_id,))
        if result :
            return {
                    "message": "删除成功",
                    "detail": f"{seach_file[0]['filename']}已经删除",
                    "status":"success"
                }
        else:
            return {
                "message": "删除失败",
                "detail": f"{len(seach_file[0]['filename'])}删除失败",
                "status": "error"
            }


    def testDb(self):

        loader = MinerULoader(source="D:/newProject/langchain_v3/uploads/e930e9aac7ec424091e5b19236df0b29.docx", mode="flash")
        #注意：这里可能会抛出 ValueError
        reslut = loader.load()
        print(reslut)
        #db.close()
        # loader = Docx2txtLoader("D:/newProject/langchain_v3/uploads/882f26b2956644e6ac5fa5e5a3efc956.docx")
        # docs = loader.load()
        # print(f"成功解析，共 {len(docs)} 个元素")

if __name__ == "__main__":
    # 测试单文件路径
    server = RagUploader()
    # 注意：这里需要提供一个实际存在的测试文件路径
    test_path = "D:/newProject/langchain_v3/uploads/4f4779fdd94040f78fa2d5e76c614d59.docx"
    #print(server.get_upload_file(test_path))
    #print(server.del_data())
    print(server.del_file())