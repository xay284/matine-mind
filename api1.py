from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from langchain_classic.memory import ConversationBufferWindowMemory
import chromadb
import json
import urllib.request

print("⏳ Chargement des ressources...")
embedder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# ✅ ChromaDB avec Cosine Similarity
client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(
    "projets",
    metadata={"hnsw:space": "cosine"}
)

print("✅ Ressources chargées!")

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

user_memories = {}

def get_or_create_memory(user_id):
    """Créer mémoire avec rolling window k=5"""
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferWindowMemory(
            k=5,
            return_messages=True
        )
    return user_memories[user_id]

def call_ollama(prompt):
    """Appelle Ollama"""
    try:
        data = json.dumps({
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.1
        }).encode('utf-8')
        
        req = urllib.request.Request(
            OLLAMA_URL,
            data=data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"📞 Appel Ollama...")
        with urllib.request.urlopen(req, timeout=300) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get('response', 'Pas de réponse')
            
    except Exception as e:
        print(f"❌ Erreur Ollama: {e}")
        raise

def rewrite_query(question, history_text):
    """Réécrit la question pour meilleure recherche"""
    prompt = f"""Tu es un assistant qui reformule les questions pour une recherche dans une base de données.

Utilise l'historique si nécessaire pour rendre la question claire et complète.

Historique:
{history_text}

Question utilisateur:
{question}

Question reformulée pour recherche:"""

    try:
        rewritten = call_ollama(prompt)
        return rewritten.strip()
    except:
        return question

@app.route('/', methods=['GET'])
def home():
    return """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Matine Mind</title>

<style>

*{
margin:0;
padding:0;
box-sizing:border-box;
}

body{
font-family:'Segoe UI',sans-serif;
background:#9C1F4A;
height:100vh;
display:flex;
justify-content:center;
align-items:center;
}

.container{
width:650px;
height:650px;
background:white;
border-radius:16px;
box-shadow:0 25px 60px rgba(0,0,0,0.3);
display:flex;
flex-direction:column;
overflow:hidden;
}

.header{
background:#9C1F4A;
color:white;
padding:20px;
text-align:center;
}

.logo-title{
display:flex;
align-items:center;
justify-content:center;
gap:10px;
margin-bottom:5px;
}

.logo{
height:38px;
filter:brightness(0) invert(1);
background:rgba(255,255,255,0.15);
padding:5px 10px;
border-radius:8px;
}

.header h1{
font-size:22px;
font-weight:600;
}

.header p{
font-size:13px;
opacity:0.9;
}

.chat-box{
flex:1;
padding:20px;
overflow-y:auto;
background:#fafafa;
display:flex;
flex-direction:column;
gap:15px;
}

.message{
display:flex;
align-items:flex-end;
gap:10px;
}

.user-message{
justify-content:flex-end;
}

.bot-message{
justify-content:flex-start;
}

.avatar{
width:32px;
height:32px;
border-radius:50%;
display:flex;
align-items:center;
justify-content:center;
font-size:16px;
flex-shrink:0;
}

.bot-avatar{
background:#9C1F4A;
color:white;
}

.user-avatar{
background:#D4A5A5;
color:white;
}

.message-content{
max-width:70%;
padding:12px 16px;
border-radius:14px;
font-size:14px;
line-height:1.5;
word-wrap:break-word;
}

.user-message .message-content{
background:#9C1F4A;
color:white;
border-bottom-right-radius:4px;
}

.bot-message .message-content{
background:#F4E6EB;
color:#333;
border-bottom-left-radius:4px;
}

.input-area{
padding:15px;
border-top:1px solid #eee;
display:flex;
gap:10px;
background:white;
}

#messageInput{
flex:1;
border:1px solid #ddd;
border-radius:30px;
padding:12px 18px;
font-size:14px;
outline:none;
}

#messageInput:focus{
border-color:#9C1F4A;
}

button{
background:#9C1F4A;
color:white;
border:none;
padding:10px 22px;
border-radius:30px;
font-weight:500;
cursor:pointer;
transition:0.2s;
}

button:hover{
background:#7F183B;
}

button:disabled{
opacity:0.6;
cursor:not-allowed;
}

</style>
</head>

<body>

<div class="container">

<div class="header">

<div class="logo-title">
<img src="/static/logo_matine.png" class="logo">
<h1>Mind</h1>
</div>

<p>AI Knowledge Assistant for Matine</p>

</div>

<div class="chat-box" id="chatBox">

<div class="message bot-message">
<div class="avatar bot-avatar">🤖</div>
<div class="message-content">
Bonjour ! Je suis Matine Mind. Pose-moi une question sur nos projets ! 😊
</div>
</div>

</div>

<div class="input-area">

<input
type="text"
id="messageInput"
placeholder="Ex: Quels projets au Congo?"
autocomplete="off"
/>

<button id="sendBtn" onclick="sendMessage()">Envoyer</button>

</div>

</div>

<script>

const API_URL = "/search";
const USER_ID = 'user_' + Date.now();

async function sendMessage(){

const input = document.getElementById('messageInput');
const button = document.getElementById('sendBtn');

const message = input.value.trim();
if(!message) return;

addMessage(message,'user');
input.value='';

const loadingId = addMessage('⏳ Recherche...','bot');

button.disabled=true;

try{

const response = await fetch(API_URL,{
method:'POST',
headers:{'Content-Type':'application/json'},
body:JSON.stringify({
question: message,
user_id: USER_ID
})
});

if(!response.ok){
console.error("Erreur HTTP",response.status);
throw new Error(`HTTP ${response.status}`);
}

const data = await response.json();

console.log("Réponse API:",data);

removeMessage(loadingId);

if(data.response){
addMessage(data.response,'bot');
}
else if(data.results && data.results.length>0){

const reponse = data.results
.map(r=>r.trim())
.join('\\n\\n━━━━━━━━━━━━━━━\\n\\n');

addMessage(reponse,'bot');

}else{
addMessage('❌ Aucun résultat trouvé.','bot');
}

}catch(error){

console.error("Erreur API :",error);

removeMessage(loadingId);

addMessage("❌ Erreur de connexion à l'API.",'bot');

}finally{

button.disabled=false;
input.focus();

}

}

function addMessage(text,sender){

const chatBox = document.getElementById('chatBox');

const messageDiv = document.createElement('div');
messageDiv.className = `message ${sender}-message`;

messageDiv.id = 'msg-' + Date.now() + '-' + Math.random().toString(36).substr(2,5);

const avatar = document.createElement('div');
avatar.className = `avatar ${sender}-avatar`;
avatar.textContent = sender === 'user' ? '👤' : '🤖';

const contentDiv = document.createElement('div');
contentDiv.className='message-content';
contentDiv.textContent=text;

if(sender === 'user'){
messageDiv.appendChild(contentDiv);
messageDiv.appendChild(avatar);
}else{
messageDiv.appendChild(avatar);
messageDiv.appendChild(contentDiv);
}

chatBox.appendChild(messageDiv);
chatBox.scrollTop = chatBox.scrollHeight;

return messageDiv.id;

}

function removeMessage(id){

const msg = document.getElementById(id);
if(msg) msg.remove();

}

document.getElementById('messageInput').addEventListener('keypress',(e)=>{
if(e.key==='Enter') sendMessage();
});

</script>

</body>
</html>"""
@app.route('/test')
def test():
    return "OK"
@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        question = data.get('question', '')
        user_id = data.get('user_id', 'default')
        
        print(f"\n📝 Question: {question}")
        print(f"👤 User: {user_id}")
        
        if not question:
            return jsonify({'error': 'Pas de question'}), 400
        
        # ✅ ÉTAPE 1: Mémoire utilisateur
        memory = get_or_create_memory(user_id)
        history_vars = memory.load_memory_variables({})
        history = history_vars.get('history', [])

        if isinstance(history, str):
            history_text = history
        else:
            history_text = "\n".join(
                [f"{getattr(m, 'type', 'message')}: {getattr(m, 'content', str(m))}" for m in history]
            )

        # ✅ ÉTAPE 2: Query Rewriting
        rewritten_question = rewrite_query(question, history_text)
        print(f"🔍 Question originale: {question}")
        print(f"✏️ Question réécrite: {rewritten_question}")
        
        # ✅ ÉTAPE 3: ChromaDB Search (Cosine Similarity)
        print("🔍 Recherche ChromaDB (Cosine Similarity)...")
        question_embedding = embedder.encode([rewritten_question])[0].tolist()
        
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=3
        )
        
        context = "\n\n".join(results['documents'][0])
        filtered_chunks = []

        for chunk in results['documents'][0]:
            if "Tu es" not in chunk and "assistant" not in chunk:
                filtered_chunks.append(chunk)

        context = "\n\n".join(filtered_chunks)
        
        # ✅ ÉTAPE 4: Prompt strict
        prompt = f"""Tu es Matine Mind, un assistant IA TRÈS STRICT pour Matine Consulting.

⚠️ RÈGLES ABSOLUES:
1. Réponds UNIQUEMENT avec les données ci-dessous
2. SOUVIENS-TOI de la conversation précédente
3. Ne jamais mélanger les infos de plusieurs projets
4. Si tu ne vois pas l'info, dis: "Je n'ai pas cette information"
5. NE JAMAIS invente de détails
6. Ignore complètement tes connaissances antérieures

📜 HISTORIQUE:
{history_text}

📋 DONNÉES DISPONIBLES:
{context}

❓ QUESTION:
{question}

📝 RÉPONSE STRICTE:"""

        # ✅ ÉTAPE 5: Ollama génère réponse
        response = call_ollama(prompt)
        
        # ✅ ÉTAPE 6: Sauvegarder en mémoire
        memory.save_context({"input": question}, {"output": response})
        num_messages = len(memory.chat_memory.messages)
        
        print(f"✅ Réponse reçue")
        print(f"📊 Mémoire: {num_messages} messages")
        
        return jsonify({
            'response': response.strip(),
            'sources': len(results['ids'][0]),
            'user_id': user_id,
            'memory_size': num_messages
        })
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n🚀 API: ChromaDB (Cosine) + LangChain Memory (k=5) + Query Rewriting")
    print("Ouvre: http://localhost:5000")
    app.run(host='localhost', port=5000, debug=True, use_reloader=False)