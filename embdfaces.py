from libraries import *

modelFN512 = DeepFace.build_model("Facenet512")
emdfaces,labels = load_embeddings("criminals", modelFN512,dist = 'cosine')


df = pd.DataFrame(emdfaces)
df.insert(0,"labels",labels)
df.to_csv("criminal.csv",index = False)
