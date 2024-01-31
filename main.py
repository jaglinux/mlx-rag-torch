import argparse
from unstructured.partition.auto import partition

def convert_pdf(pdf):
    elements = partition(filename=pdf)
    #print("\n\n".join([str(el) for el in elements]))
    return "\n\n".join([el.text for el in elements])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="pytorch-rag-pdf",
                            description="connvert pdf data to vector DB, infer by using torch")
    parser.add_argument("-f", "--file",
                        help="Provide pdf file name, should be present in current directory",
                        default="flash_attention.pdf")
    args = parser.parse_args()
    pdf = args.file
    elements = convert_pdf(pdf)