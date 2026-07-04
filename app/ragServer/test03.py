import os
import smtplib
import imaplib
import email
from email.mime.text import MIMEText
from email.header import decode_header
from dotenv import load_dotenv
from langchain.tools import tool
from email.utils import parseaddr
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
load_dotenv()


# 1. 发送邮件工具
@tool
def send_qq_email(to: str, subject: str, body: str) -> str:
    """
    通过 QQ 邮箱发送邮件给指定收件人。
    Args:
        to: 收件人邮箱地址
        subject: 邮件主题
        body: 邮件正文内容
    """
    try:
        msg = MIMEText(body, 'plain', 'utf-8')
        msg['From'] = os.getenv("QQ_EMAIL")
        msg['To'] = to
        msg['Subject'] = subject

        # 使用 SSL 方式连接 QQ 邮箱 SMTP 服务器 (端口 465)
        server = smtplib.SMTP_SSL("smtp.qq.com", 465)
        server.login(os.getenv("QQ_EMAIL"), os.getenv("QQ_PASSWORD"))
        server.sendmail(os.getenv("QQ_EMAIL"), to, msg.as_string())
        server.quit()
        return f"✅ 邮件已成功发送至 {to}"
    except Exception as e:
        return f"❌ 发送邮件失败: {str(e)}"


# 2. 读取邮件工具
@tool
def read_qq_emails(num_emails: int = 5) -> list[dict]:
    """
    读取 QQ 邮箱收件箱中最近的邮件。
    Args:
        num_emails: 需要获取的最新邮件数量，默认5封。
    """
    try:
        mail = imaplib.IMAP4_SSL("imap.qq.com")
        mail.login(os.getenv("QQ_EMAIL"), os.getenv("QQ_PASSWORD"))
        mail.select("INBOX")


        status, messages = mail.search(None, "ALL")
        email_ids = messages[0].split()

        # 获取最新的 num_emails 封邮件
        recent_ids = email_ids[-num_emails:]
        emails_data = []

        for e_id in recent_ids:
            #print(f"打印e_id:{e_id}")
            status, msg_data = mail.fetch(e_id, "(RFC822)")

            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)

            #解码邮件的正文
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    payload = part.get_payload(decode=True)  # 自动进行 Base64 解码
                    charset = part.get_content_charset() or 'utf-8'
                    body_content = payload.decode(charset, errors='ignore')


            # 解码邮件主题
            subject, encoding = decode_header(msg["Subject"])[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding if encoding else "utf-8")

            #解码邮件来源from
            # 解码 From 字段
            raw_from = msg.get("From")
            decoded_parts = decode_header(raw_from)

            # 拼接解码后的字符串
            from_name = ""
            for part, charset in decoded_parts:
                if isinstance(part, bytes):
                    from_name += part.decode(charset or 'utf-8', errors='ignore')
                else:
                    from_name += part
            _, pure_email = parseaddr(from_name)

            emails_data.append({
                "subject": subject,
                "from": pure_email,
                "content":body_content
            })

        mail.close()
        mail.logout()
        print(emails_data)
        return emails_data
    except Exception as e:
        return [{"error": f"读取邮件失败: {str(e)}"}]



system_prompt = ("你是一个智能邮件助手，可以帮用户读取和发送QQ邮箱邮件。"
                 "1.读取邮件的时候，用工具read_qq_emails获取邮件，会返回：subject，from，content；"
                 "2.当用户回复邮件的时候，用工具send_qq_email ，用户提示词会包含邮件的收件人，主题，内容，"
                 "例如：给1048079943@qq.com回复一封邮件，主题是今天晚上有很多萤火虫，内容为今天放假了，大家都在一起露营，"
                 "晚上草丛中有一大堆萤火虫，十分漂亮")

model =init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)


agent = create_agent(
            model=model,
            tools=[read_qq_emails],
            system_prompt=system_prompt,
)


# 测试调用
param = agent.invoke({
    "messages": [
        {"role": "user", "content": "读取邮件"}
    ]
})
print(param)