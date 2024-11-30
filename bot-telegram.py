import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

#env\Scripts\activate

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_LABELS = 2

tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
model = AutoModel.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')

class Classifier(nn.Module):
    def __init__(self, model, num_labels):
        super(Classifier, self).__init__()
        self.base_model = model
        self.cls = nn.Linear(model.config.hidden_size, 400)
        self.dropout = nn.Dropout(p=0.5)
        self.cls2 = nn.Linear(400, num_labels)
        self.gelu = nn.GELU()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0][:, 0, :] 
        prediction = self.cls(sequence_output)
        prediction = self.gelu(prediction)
        prediction = self.dropout(prediction)
        prediction = self.cls2(prediction)
        return prediction

#carregar modelo
modelo_path = "model-roberta.pth"  
model_roberta = Classifier(model, NUM_LABELS).to(DEVICE)
model_roberta.load_state_dict(torch.load(modelo_path, map_location=DEVICE))
model_roberta.eval()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Me diga o que está sentindo e eu classificarei como 'urgente' ou 'não urgente'.")

def classificar_texto(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        outputs = model_roberta(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        predicted_label = torch.argmax(outputs, dim=-1).item()
    return "urgente" if predicted_label == 1 else "não urgente"

async def identificar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    texto = update.message.text
    label = classificar_texto([texto])
    await update.message.reply_text(f"A sua atual situação é: {label}")

def main():
    app = Application.builder().token("TOKEN").build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, identificar))

    app.run_polling()

if __name__ == '__main__':
    main()
