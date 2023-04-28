import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `En utilisant la conversation suivante et une question de suivi, reformulez la question de suivi pour en faire une question autonome.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `Vous êtes un assistant virtuel utile. Utilisez les éléments de contexte suivants pour répondre à la question posée à la fin.
Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas. NE PAS essayer de faire une réponse inventée.
Si la question n'est pas liée au contexte, répondez poliment que vous êtes programmé pour répondre uniquement aux questions liées au contexte.

{context}

Question: {question}
Helpful answer in markdown:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0, // increase temepreature to get more creative answers
    modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
