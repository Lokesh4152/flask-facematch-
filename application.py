from cProfile import label
from calendar import month
from email.mime import application
from flask import Flask, request,jsonify
from libraries import *
from os.path import exists 


application = Flask(__name__)


@application.route("/",methods=["GET"])
def hello():
    return "This is the first page :"

@application.route("/api/matchface",methods=["GET","POST"])
def facematch():
    if request.method == "GET":
        return jsonify({'response': "Get Request Called"})
    elif request.method =='POST':
        req_Json = request.json
        imgUrl = str(req_Json['imgUrl'])
        url = imgUrl
        # count = random.randint(0,9999999)
        # file_name = "img"+str(count)+".jpg" #prompt user for file_name
        res = requests.get(url, stream = True)
        aurl = url.split('/')

        aurl = aurl[4]

        aurl = aurl.partition(".")[0]
        aurl = aurl + ".jpg"
        file_exists = exists(aurl)
        file_name = aurl
        if(file_exists):
            pass
        else:
            if res.status_code == 200:
                with open(file_name,'wb') as f:
                    shutil.copyfileobj(res.raw, f)
                print('Image sucessfully Downloaded: ',file_name)

            else:
                print('Image Couldn\'t be retrieved')







        # if res.status_code == 200:
        #     with open(file_name,'wb') as f:
        #         shutil.copyfileobj(res.raw, f)
        #     print('Image sucessfully Downloaded: ',file_name)
        # else:
        #     print('Image Couldn\'t be retrieved')

        modelFN512 = DeepFace.build_model("Facenet512")
        ppEmbd = get_embedding(modelFN512, file_name)
        dfcrmnl = pd.read_csv("criminal.csv")
        emdfaces = dfcrmnl.iloc[:,1:]
        emdfaces=np.asarray(emdfaces)
        labels = dfcrmnl.iloc[:,0]
        d =[]
        for e in emdfaces:
            d.append(findCosineDistance(e,ppEmbd))
        faceIndex = np.argmin(d)    

        similarity_score= 1 - d[faceIndex]
        if similarity_score > 0.7:
            msg = f"The test Image is of: {labels[faceIndex]} Matching with {str(1-d[faceIndex])} Similarity Score"
            return jsonify({"Resonse:": msg })
        else:
            return jsonify({"Response:": "The test Image matches none"})



if __name__ == "__main__":
    application.run(debug = True,port = 8000)
    