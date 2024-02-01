import argparse
from unstructured.partition.auto import partition
from embedding_model_gte_large import Embedding_Model, caclulate_scores_index

def convert_pdf(pdf):
    elements = partition(filename=pdf)
    #print("\n\n".join([str(el) for el in elements]))
    return "\n\n".join([el.text for el in elements])

def convert_to_chunks(elements):
    start, end = 0,0
    chunk_size = 1000
    overlap = 200
    chunks= []
    while start < len(elements):
        end = start + chunk_size
        chunks.append(elements[start:end])
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="pytorch-rag-pdf",
                            description="connvert pdf data to vector DB, infer by using torch")
    parser.add_argument("-f", "--file",
                        help="Provide pdf file name, should be present in current directory",
                        default="flash_attention.pdf")
    args = parser.parse_args()
    pdf = args.file
    elements = convert_pdf(pdf)
    chunks = convert_to_chunks(elements)
    query = "Standard Attention Implementation"
    e = Embedding_Model()
    source_data_emb = e.create_embeddings(chunks)
    query_data_emb = e.create_embeddings(query)
    index = caclulate_scores_index(source_data_emb, query_data_emb)
    print(f"question is {query}")
    print(f"index is {index} and Context is {chunks[index]}")