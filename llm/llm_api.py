from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()  # ✅ Define FastAPI app first

# Enable CORS to prevent errors when calling API from a browser or different origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.post("/generate")
async def generate_text(data: dict):
    prompt = data.get("prompt", "")
    output = generator(prompt, max_length=150, do_sample=True, top_p=0.9, temperature=0.7)
    return {"response": output[0]['generated_text']}

@app.get("/")
def read_root():
    return {"message": "FastAPI server is running!"}
