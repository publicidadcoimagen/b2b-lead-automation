"""
Sistema de Clasificación de Leads con IA
Automatización inteligente de adquisición B2B

Requisitos:
pip install pandas openai python-dotenv requests
"""

import pandas as pd
import json
import os
from datetime import datetime
from dotenv import load_dotenv
import openai

load_dotenv()

# Configurar OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")


class LeadScoringSystem:
    """
    Sistema automatizado de puntuación y clasificación de leads.
    Utiliza IA para evaluar probabilidad de cierre y generar respuestas personalizadas.
    """
    
    def __init__(self, input_csv_path: str):
        """
        Inicializa el sistema con un archivo CSV de leads.
        
        Args:
            input_csv_path: Ruta al archivo CSV con leads
        """
        self.input_csv = input_csv_path
        self.leads_df = None
        self.scored_leads = []
        self.load_leads()
    
    def load_leads(self):
        """Carga los leads desde CSV."""
        try:
            self.leads_df = pd.read_csv(self.input_csv)
            print(f"✅ Cargados {len(self.leads_df)} leads desde {self.input_csv}")
            print(f"   Columnas: {list(self.leads_df.columns)}\n")
        except FileNotFoundError:
            print(f"❌ Archivo no encontrado: {self.input_csv}")
            raise
    
    def score_lead(self, lead_data: dict) -> dict:
        """
        Clasifica un lead usando IA.
        
        Args:
            lead_data: Diccionario con información del lead
            
        Returns:
            Diccionario con score y clasificación
        """
        
        prompt = f"""
        Eres un experto en clasificación de leads B2B. 
        Analiza este lead y proporciona:
        
        1. Score de compra (0-100)
        2. Clasificación: HOT / WARM / COLD
        3. Razones principales
        4. Próximo mejor paso
        
        INFORMACIÓN DEL LEAD:
        {json.dumps(lead_data, indent=2, ensure_ascii=False)}
        
        RESPONDE EN JSON:
        {{
            "score": <0-100>,
            "classification": "<HOT/WARM/COLD>",
            "reasons": ["razón1", "razón2"],
            "next_action": "<próximo paso>",
            "confidence": "<ALTA/MEDIA/BAJA>"
        }}
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            result_text = response.choices[0].message.content
            score_data = json.loads(result_text)
            return score_data
            
        except json.JSONDecodeError:
            print(f"⚠️  Error parseando respuesta para {lead_data.get('email')}")
            return {
                "score": 50,
                "classification": "WARM",
                "reasons": ["No se pudo clasificar"],
                "next_action": "Revisar manualmente",
                "confidence": "BAJA"
            }
        except Exception as e:
            print(f"❌ Error en IA: {str(e)}")
            raise
    
    def process_all_leads(self):
        """
        Procesa todos los leads y genera scores.
        """
        print("🚀 Iniciando procesamiento de leads...\n")
        
        for idx, row in self.leads_df.iterrows():
            lead_data = row.to_dict()
            
            print(f"[{idx+1}/{len(self.leads_df)}] Clasificando: {lead_data.get('email')}")
            
            score_info = self.score_lead(lead_data)
            
            result = {
                **lead_data,
                "lead_score": score_info["score"],
                "classification": score_info["classification"],
                "reasons": " | ".join(score_info["reasons"]),
                "next_action": score_info["next_action"],
                "confidence": score_info["confidence"],
                "processed_at": datetime.now().isoformat()
            }
            
            self.scored_leads.append(result)
            print(f"   ✓ Score: {score_info['score']}/100 - {score_info['classification']}\n")
        
        print(f"✅ Procesados {len(self.scored_leads)} leads\n")
    
    def export_results(self, output_csv="scored_leads.csv"):
        """
        Exporta los resultados a CSV.
        
        Args:
            output_csv: Nombre del archivo de salida
        """
        if not self.scored_leads:
            print("⚠️  No hay leads procesados")
            return
        
        results_df = pd.DataFrame(self.scored_leads)
        results_df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"✅ Resultados exportados a: {output_csv}")
        
        print("\n📊 ESTADÍSTICAS:")
        hot = len(results_df[results_df['classification'] == 'HOT'])
        warm = len(results_df[results_df['classification'] == 'WARM'])
        cold = len(results_df[results_df['classification'] == 'COLD'])
        print(f"   HOT:  {hot}")
        print(f"   WARM: {warm}")
        print(f"   COLD: {cold}")
        print(f"   Score promedio: {results_df['lead_score'].mean():.1f}")


def main():
    """Función principal para ejecutar el sistema."""
    
    INPUT_CSV = "leads.csv"
    OUTPUT_CSV = "scored_leads_final.csv"
    
    print("\n" + "="*60)
    print("🤖 SISTEMA DE SCORING DE LEADS CON IA")
    print("="*60 + "\n")
    
    system = LeadScoringSystem(INPUT_CSV)
    system.process_all_leads()
    system.export_results(OUTPUT_CSV)


if __name__ == "__main__":
    main()