from config import model_name,base_url
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
GLM_API = os.getenv('GLM_API_KEY')

def load_model():
    model=ChatOpenAI(model=model_name,openai_api_key=GLM_API,base_url=base_url)
    return model

def load_rewritten_prompt():
    prompt = PromptTemplate.from_template(
        """
            You are a query rewriting engine for a Retrieval-Augmented Generation (RAG) system
            with structured metadata filtering.

            The vector database stores drug information with metadata fields:

            Metadata fields:
            - generic_name: drug name
            - section: document section name

            Allowed section values:
            spl product data elements
            active ingredient
            purpose
            indications and usage
            warnings
            do not use
            stop use
            pregnancy or breast feeding
            keep out of reach of children
            dosage and administration
            storage and handling
            inactive ingredient
            package label principal display panel
            id
            recent major changes
            spl unclassified section
            description
            clinical pharmacology
            contraindications
            precautions
            adverse reactions
            overdosage
            how supplied
            when using
            mechanism of action
            pharmacokinetics
            pharmacodynamics
            general precautions
            laboratory tests
            drug interactions
            carcinogenesis and mutagenesis and impairment of fertility
            pregnancy
            teratogenic effects
            nursing mothers
            pediatric use
            geriatric use
            questions
            information for patients
            drug and or laboratory test interactions
            labor and delivery
            drug abuse and dependence
            controlled substance
            dependence
            references
            spl patient package insert
            boxed warning
            dosage forms and strengths
            warnings and cautions
            use in specific populations
            nonclinical toxicology
            clinical studies
            abuse
            instructions for use
            ask doctor
            spl medguide
            ask doctor or pharmacist
            microbiology
            other safety information
            pharmacogenomics
            animal pharmacology and or toxicology
            nonteratogenic effects
            spl indexing data elements
            patient medication information
            health care provider letter
            information for owners or caregivers
            user safety warnings
            risks
            safe handling warning
            statement of identity
            health claim
            components
            alarms
            cleaning
            disposal and waste handling


            Your task:
            Rewrite the user question into a concise, retrieval-optimized search query that aligns with these metadata fields.

            Rules:
            - Preserve original intent exactly.
            - ALWAYS include the generic drug name(s).
            - ALWAYS include the relevant section name if implied.
            - If multiple drugs are mentioned, treat them as a combination — do NOT separate them.
            - Prefer exact section wording from the allowed section list.
            - Use concise technical noun phrases.
            - Remove conversational words.
            - Do NOT answer the question.
            - Output ONLY the rewritten query.

            Examples:

            Q: What is the dosage of amoxicillin?
            A: amoxicillin dosage and administration

            Q: What are the side effects of ibuprofen?
            A: ibuprofen adverse reactions

            Q: Can aspirin and clopidogrel be used together?
            A: aspirin and clopidogrel drug interactions

            Q: How does metformin work?
            A: metformin mechanism of action

            Q: what are the mechanism of action for ensulizole,octisalate?
            A: ensulizole,octisalate mechanism of action

            User question:
            {query}

            Rewritten search query:
            """
    )
    return prompt


def load_AI_prompt():
    prompt = PromptTemplate.from_template(
        """"You are a question-answering system in a Retrieval-Augmented Generation (RAG) pipeline.
        Use ONLY the information provided in the context below to answer the question.
        Do NOT use prior knowledge or assumptions.

        If the answer cannot be found explicitly in the context, reply exactly:
        "I don't know based on the provided context."

        Context:{context}
        Question:{question}
        Answer:  """
    )
    return prompt
