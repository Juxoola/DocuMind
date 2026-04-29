import os
import chainlit as cl
from chainlit.input_widget import Switch
from src.ingestion import ingest_file
from src.rag_pipeline import build_index, get_query_engine
import config

async def update_sidebar_settings():
    """Обновляет левое меню (Chat Settings) со списком файлов."""
    files = cl.user_session.get("uploaded_files", [])
    if not files:
        return
        
    inputs = []
    for f in files:
        inputs.append(Switch(id=f, label=f, initial=True))
        
    settings = await cl.ChatSettings(inputs).send()
    cl.user_session.set("chat_settings", settings)

@cl.on_settings_update
async def setup_agent(settings):
    """Вызывается при переключении тумблеров в левом меню."""
    cl.user_session.set("chat_settings", settings)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("uploaded_files", [])
    cl.user_session.set("chat_settings", {})
    
    await cl.Message(content="Привет! Я ваш локальный аналог NotebookLM.\n\nЗагружайте видео, аудио, презентации и PDF, а затем задавайте по ним вопросы! Слева в меню вы сможете включать и выключать определенные документы для поиска.").send()
    await request_file()

async def request_file():
    files = await cl.AskFileMessage(
        content="Пожалуйста, загрузите документ или видео для добавления в базу знаний:",
        accept={
            "text/plain": [".txt"],
            "application/pdf": [".pdf"],
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": [".pptx"],
            "video/mp4": [".mp4", ".avi", ".mkv"],
            "audio/mpeg": [".mp3", ".wav", ".m4a"]
        },
        max_size_mb=5000,
        timeout=3600
    ).send()

    if files:
        file = files[0]
        msg = cl.Message(content=f"Обработка файла `{file.name}`. Пожалуйста, подождите...")
        await msg.send()
        
        file_path = os.path.join(config.DATA_DIR, file.name)
        with open(file_path, "wb") as f:
            with open(file.path, "rb") as source_f:
                f.write(source_f.read())
                
        try:
            nodes = ingest_file(file_path)
            build_index(nodes)
            
            uploaded_files = cl.user_session.get("uploaded_files", [])
            if file.name not in uploaded_files:
                uploaded_files.append(file.name)
                cl.user_session.set("uploaded_files", uploaded_files)
                
            await update_sidebar_settings()
            
            msg.content = f"Файл `{file.name}` успешно обработан! Слева в меню (значок настроек) появился переключатель для него. Вы можете задавать вопросы или загрузить еще файлы."
            await msg.update()
        except Exception as e:
            msg.content = f"Произошла ошибка при обработке: {e}"
            await msg.update()

@cl.on_message
async def main(message: cl.Message):
    settings = cl.user_session.get("chat_settings", {})
    uploaded_files = cl.user_session.get("uploaded_files", [])
    
    if not uploaded_files:
        await cl.Message(content="База пуста! Пожалуйста, сначала загрузите файлы (обновите страницу).").send()
        return

    allowed_files = []
    for f in uploaded_files:
        if settings.get(f, True):
            allowed_files.append(f)
            
    if not allowed_files:
        await cl.Message(content="Все источники выключены в левом меню! Пожалуйста, включите хотя бы один файл для поиска.").send()
        return
            
    query_engine = get_query_engine(allowed_files=allowed_files)
    
    msg = cl.Message(content="Ищу в выбранных источниках...")
    await msg.send()
    
    try:
        response = query_engine.query(message.content)
        msg.content = str(response)
        
        elements = []
        source_images = set()
        
        # Добавляем цитаты и изображения
        for idx, source_node in enumerate(response.source_nodes):
            metadata = source_node.node.metadata
            file_name = metadata.get("file_name", "Неизвестный источник")
            text_snippet = source_node.node.get_content()
            
            element_name = f"Источник {idx+1} ({file_name})"
            elements.append(cl.Text(content=text_snippet, name=element_name, display="inline"))
            
            if "image_path" in metadata:
                img_path = metadata["image_path"]
                if img_path not in source_images and os.path.exists(img_path):
                    source_images.add(img_path)
                    elements.append(cl.Image(path=img_path, name=f"Изображение из {file_name}", display="inline"))
                    
        msg.elements = elements
        await msg.update()
    except Exception as e:
        msg.content = f"Ошибка поиска: {e}"
        await msg.update()
