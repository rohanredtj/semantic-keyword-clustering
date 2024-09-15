import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import time

class SemanticKeywordClustering:
    def __init__(self, input_file, transformer='msmarco-distilbert-base-tas-b', 
                 cluster_accuracy=0.95, min_cluster_size=2):
        self.input_file = input_file
        self.transformer = transformer
        self.cluster_accuracy = cluster_accuracy
        self.min_cluster_size = min_cluster_size
        self.df = None
        self.model = None

    def load_data(self):
        self.df = pd.read_csv(self.input_file, encoding='utf-8', on_bad_lines='skip')
        self.df['Keyword'] = self.df['Keyword'].str.strip()
        return self.df.shape[0]

    def preprocess_keywords(self):
        def clean_text(text):
            text = re.sub("\d+", "", text)
            return text.strip().lower()

        self.df['Keyword'] = self.df['Keyword'].apply(clean_text)
        self.df = self.df[['Keyword']].drop_duplicates()

    def initialize_model(self):
        self.model = SentenceTransformer(self.transformer)

    def cluster_keywords(self):
        corpus_set = set(self.df['Keyword'])
        cluster_name_list = []
        corpus_sentences_list = []
        df_all = []

        while True:
            corpus_sentences = list(corpus_set)
            check_len = len(corpus_sentences)

            corpus_embeddings = self.model.encode(corpus_sentences, batch_size=256, 
                                                  show_progress_bar=True, convert_to_tensor=True)
            clusters = util.community_detection(corpus_embeddings, min_community_size=self.min_cluster_size, 
                                                threshold=self.cluster_accuracy, init_max_size=len(corpus_embeddings))

            for keyword, cluster in enumerate(clusters):
                cluster_name = f"Cluster {keyword + 1}, #{len(cluster)} Elements"
                for sentence_id in cluster:
                    corpus_sentences_list.append(corpus_sentences[sentence_id])
                    cluster_name_list.append(cluster_name)

            df_new = pd.DataFrame({'Cluster Name': cluster_name_list, 'Keyword': corpus_sentences_list})
            df_all.append(df_new)
            
            corpus_set -= set(df_new['Keyword'])
            remaining = len(corpus_set)
            print(f"Total Unclustered Keywords: {remaining}")
            
            if check_len == remaining:
                break

        return pd.concat(df_all), remaining

    def postprocess_results(self, df_clustered, total_keywords):
        self.df = self.df.merge(df_clustered.drop_duplicates('Keyword'), how='left', on="Keyword")
        self.df['Cluster Name'] = self.df.groupby('Cluster Name')['Keyword'].transform('first')
        self.df['Cluster Name'] = self.df['Cluster Name'].fillna("zzz_no_cluster")
        self.df = self.df.sort_values(["Cluster Name", "Keyword"])

        clustered_percent = ((total_keywords - df_clustered['Keyword'].nunique()) / total_keywords) * 100
        print(f"{clustered_percent:.2f}% of rows clustered successfully!")

    def run(self):
        start_time = time.time()

        print("Loading data...")
        total_keywords = self.load_data()

        print("Preprocessing keywords...")
        self.preprocess_keywords()

        print("Initializing model...")
        self.initialize_model()

        print("Clustering keywords...")
        df_clustered, remaining = self.cluster_keywords()

        print("Postprocessing results...")
        self.postprocess_results(df_clustered, total_keywords)

        print(f"Total execution time: {time.time() - start_time:.2f} seconds")

        return self.df

if __name__ == "__main__":
    input_file = "sample_input_file.csv"
    output_file = "output_file.csv"

    clustering = SemanticKeywordClustering(input_file)
    result_df = clustering.run()
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
