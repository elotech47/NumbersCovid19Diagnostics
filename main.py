import streamlit as st  
#from gradcam_02 import GradCAM
from gradCam import GradCAM
import cv2
from PIL import Image, ImageEnhance
import numpy as np 
import pandas as pd 
import os
import csv
import keras
from tensorflow.keras.models import load_model
from imutils import paths
import imutils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
# from keras.utils import CustomObjectScope
# from keras.initializers import glorot_uniform
from keras import backend as K
import matplotlib.pyplot as plt

with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def getList(dict): 
    return [*dict] 
@st.cache(allow_output_mutation=True)
def loadModel():
    # with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    #     model = load_model('covid19_model3.h5' )#, compile=False)
    model = load_model('covid19_model3.h5' )
    print("Model Loaded Succesfully")
    print(model.summary())
    return model

#@st.cache(allow_output_mutation=True)
def Diagnose(image):
    global label
    #model,session = model_upload()
    model= loadModel()
    image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, ( 224, 224))
    cv2.imshow("image", image)
    data = []
    data.append(image)
    # convert the data and labels to NumPy arrays while scaling the pixel
    # intensities to the range [0, 255]
    data = np.array(data) / 255.0
    #K.set_session(session)
    pred = model.predict(data)
    return pred
   

    # st.write(alt.Chart(df).mark_bar().encode(
    # x=alt.X('Status', sort=None),
    # y='Percentage',))

def main():

    """An AI Diagnostic app for detecting Covid-19 from X-ray Scan images"""
    #image
    from PIL import Image
    img = Image.open("Numbers Logo.png")
    st.image(img, width=150,caption="")
    st.title("COVID AI DIAGNOSTICS")
    st.title("Numb3rs AI-Based Covid-19 Diagnostics")
    st.text("Built by Numb3rs input and output technology")

    activities = ["upload", "Questions", "Welcome", "About"]
    choice = st.sidebar.selectbox("Select Activty",activities)

    if choice == "Welcome":
        st.subheader("Welcome to Numb3rs AI DIagnostic Tool")
        img = Image.open("bgImage.jpg")
        st.image(img, width=600,caption="Deeplearning based Screening")
        st.write("Numb3rs AI is a computer program that uses Deep Neural Network to diagnose for COVID-19 using X_ray images of patients")
        st.write("One of the major issues we have combacting the corona virus pandemic is the slow testing process")
        st.write("Hence we created this easy to use deep learning program with 97 percent testing accuracy.")
        st.write( "Numb3rs AI would support doctors when detecting this virus hence, avoiding congestion in hospitals, clinics and testing health centers")
        #st.write("This programs uses Deep Transfer learning and X-ray images to detect Covid-19 from chest X-ray radiograph.")
        st.write("")
        st.write("")        
        st.subheader("User guide")
        st.write("Select Upload from the select activity dropdown from the left side of the page")
        st.write("Upload the X-ray image taken of the patient")
        st.write("Click on Diagnose")
        st.write("The prediction comes with percentage of certainty and a heatmap that shows the areas in the X-ray Affected")

    if choice == 'upload':
        st.subheader("Upload X-ray Image")
        image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])
        
        if image_file is not None:
            Sampleimage = Image.open(image_file)
            st.text("X_ray Image")
            st.success("Successful loaded X-ray Image")
            is_check = st.checkbox("Display Image")
            if is_check:
                st.image(Sampleimage,width=300)

        name = st.text_input("Enter name of patient-diagnosis")
        if st.button("Diagnose"):
            pred = Diagnose(Sampleimage)
            i = np.argmax(pred[0])
            covid = pred[0][0]
            normal = pred[0][1]
            data = (np.around([covid, normal],decimals = 2))*100
            covidR = data[0]
            normalR = data[1]
            st.info("Here is the Diagnosis")
            if covid >= normal:
                imageID, label, prob = [1, "covid19", covid*100]
                st.write("Covid-19 Suspected with {} certainty".format(prob))
                label = "{}: {:.2f}%".format(label, prob )
                st.write("[INFO] {}".format(label))
            else:
                imageID, label, prob = [1, "normal", normal*100]
                st.write("Normal Condition Suspected with {} certainty".format(prob))
                label = "{}: {:.2f}%".format(label, prob)
                st.write("[INFO] {}".format(label))
            my_dict = {"covid":covidR,"normal":normalR}
            df = pd.DataFrame(list(my_dict.items()),columns = ['Status','Percentage']) 
                # Get a color map
            my_cmap = cm.get_cmap('jet')
            
            # Get normalize function (takes data in range [vmin, vmax] -> [0, 1])
            my_norm = Normalize(vmin=0, vmax=100)
            plt.bar("Status", "Percentage", data = df, color = my_cmap(my_norm(data)))
            plt.xlabel("Status")
            plt.ylabel("Percentage")
            plt.title("Percentage of Status")
            st.pyplot()
            
            if name == None:
                st.error("Please fill in patient name and diagnosis seperated with {}".format("-"))
            else:
                name_save = "{}.jpeg".format(name)
                nameDetailed = "{}_detailed.jpeg".format(name)
                model = loadModel()
                cam = GradCAM(model, i)
                Reimage = cv2.cvtColor(np.float32(Sampleimage), cv2.COLOR_BGR2RGB)
                Reimage = cv2.resize(Reimage, ( 224, 224))
                ReimageGrad = np.array(Reimage)/255.0
                ReimageGrad = np.expand_dims(ReimageGrad, axis=0)
                imageArray = np.array(Sampleimage)
                orig = cv2.cvtColor(imageArray, cv2.COLOR_BGR2RGB)
                orig =cv2.resize(orig, ( 224, 224))
                cam_image = ReimageGrad#cv2.imread(imagepath)
                gradImageOutput = cam.compute_heatmap(cam_image)#, label, name)
                #st.write("[INFO]: Showing GradCam Heatmap") # and then overlay heatmap on top of the image
                st.success("Check below for Detailed Diagnosis analysis")
                heatmap = cv2.resize(gradImageOutput, (orig.shape[1], orig.shape[0]))
                (heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

                #draw the predicted label on the output image
                cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
                cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
                # display the original image and resulting heatmap and output image
                # to our screen
                cv2.imwrite(name_save,output)
                outputCompare = np.hstack([orig, heatmap, output])
                outputCompare = imutils.resize(outputCompare, height=900, width = 900)
                #cv2.imshow("Output", outputCompare)
                cv2.imwrite(nameDetailed,outputCompare)
                #if st.button("Investigate"):
                explainedImage = Image.open(nameDetailed)
                st.image(explainedImage,width=900)
                #st.write("investigated")
                #cv2.waitKey(0)

        st.subheader("Upload X-ray Image Results")
        if st.button("Compare Images"):
            positive_image = st.file_uploader("Upload Positive Result",type=['jpg','png','jpeg'])
            if positive_image is not None:
                positive_image = Image.open(positive_image)
                st.success("Successful loaded positive X-ray Image")
                #is_check = st.checkbox("Display Image")
                st.image(positive_image,width=500)
            negative_image = st.file_uploader("Upload Negative Result",type=['jpg','png','jpeg'])
            if negative_image is not None:
                negative_image = Image.open(negative_image)
                st.success("Successful loaded Negative X-ray Image")
                #is_check = st.checkbox("Display Image")
                #if is_check:
                st.image(negative_image,width=500)

            positive_image = np.array(positive_image)
            positive_image = cv2.cvtColor(positive_image, cv2.COLOR_BGR2RGB)
            positive_image = cv2.resize(positive_image, ( 224, 224))

            negative_image = np.array(negative_image)
            negative_image = cv2.cvtColor(negative_image, cv2.COLOR_BGR2RGB)
            negative_image = cv2.resize(negative_image, ( 224, 224))

            outputCompareXray = np.vstack([positive_image, negative_image])
            outputCompareXray = imutils.resize(outputCompareXray, height=180, width = 640)

            cv2.imwrite("ComparedImages.jpeg",outputCompareXray)
            #if st.button("Investigate"):
            explainedXray = Image.open("ComparedImages.jpeg")
            st.image(explainedXray)#,width=300)


        #st.write(name)
        #label  = st.radio("What is the status", ("COVID19", "Normal"))
    if choice == "About":
        st.subheader("About COVID AI DIAGNOSTICS")
        st.write("This programs uses Deep Transfer learning and X-ray images to detect Covid-19 from chest X-ray radiograph.")
        img2 = Image.open("download.jpg")
        st.text("VGG16 Architecture")
        st.image(img2, width=500,caption="Photo Credit: https://www.researchgate.net/figure/Fig-A1-The-standard-VGG-16-network-architecture-as-proposed-in-32-Note-that-only_fig3_322512435")
    
    if choice == "Questions":
        st.header("Please Answer the following questions for detailed investigations")
        fullname = "fullname"
        fullname = st.text_input("Enter Your Full Name", "Enter Here...")
        fullname = fullname.title()
        age = []
        age = st.text_input("Enter your Age", "Enter Age Here")
        age = age.title()
        phone = st.text_input("Enter your Phone Number", "Enter Here")
        phone = phone.title()
        
        LGA = { "Abia":(
                    "Aba North",
                    "Aba South",
                    "Arochukwu",
                    "Bende",
                    "Ikawuno",
                    "Ikwuano",
                    "Isiala-Ngwa North",
                    "Isiala-Ngwa South",
                    "Isuikwuato",
                    "Umu Nneochi",
                    "Obi Ngwa",
                    "Obioma Ngwa",
                    "Ohafia",
                    "Ohaozara",
                    "Osisioma",
                    "Ugwunagbo",
                    "Ukwa West",
                    "Ukwa East",
                    "Umuahia North",
                    "Umuahia South"
                ),
        "Adamawa":(
                    "Demsa",
                    "Fufore",
                    "Ganye",
                    "Girei",
                    "Gombi",
                    "Guyuk",
                    "Hong",
                    "Jada",
                    "Lamurde",
                    "Madagali",
                    "Maiha",
                    "Mayo-Belwa",
                    "Michika",
                    "Mubi-North",
                    "Mubi-South",
                    "Numan",
                    "Shelleng",
                    "Song",
                    "Toungo",
                    "Yola North",
                    "Yola South"
                ),
            "Akwa Ibom":
                (
                    "Abak",
                    "Eastern-Obolo",
                    "Eket",
                    "Esit-Eket",
                    "Essien-Udim",
                    "Etim-Ekpo",
                    "Etinan",
                    "Ibeno",
                    "Ibesikpo-Asutan",
                    "Ibiono-Ibom",
                    "Ika",
                    "Ikono",
                    "Ikot-Abasi",
                    "Ikot-Ekpene",
                    "Ini",
                    "Itu",
                    "Mbo",
                    "Mkpat-Enin",
                    "Nsit-Atai",
                    "Nsit-Ibom",
                    "Nsit-Ubium",
                    "Obot-Akara",
                    "Okobo",
                    "Onna",
                    "Oron",
                    "Oruk Anam",
                    "Udung-Uko",
                    "Ukanafun",
                    "Urue-Offong/Oruko",
                    "Uruan",
                    "Uyo"),"Anambra":
                    ("Aguata",
                    "Anambra East",
                    "Anambra West",
                    "Anaocha",
                    "Awka North",
                    "Awka South",
                    "Ayamelum",
                    "Dunukofia",
                    "Ekwusigo",
                    "Idemili-North",
                    "Idemili-South",
                    "Ihiala",
                    "Njikoka",
                    "Nnewi-North",
                    "Nnewi-South",
                    "Ogbaru",
                    "Onitsha-North",
                    "Onitsha-South",
                    "Orumba-North",
                    "Orumba-South"
                ),
        "Bauchi":(
                    "Alkaleri",
                    "Bauchi",
                    "Bogoro",
                    "Damban",
                    "Darazo",
                    "Dass",
                    "Gamawa",
                    "Ganjuwa",
                    "Giade",
                    "Itas\/Gadau",
                    "Jama'Are",
                    "Katagum",
                    "Kirfi",
                    "Misau",
                    "Ningi",
                    "Shira",
                    "Tafawa-Balewa",
                    "Toro",
                    "Warji",
                    "Zaki"
                ),
        "Benue":(
                    "Ado",
                    "Agatu",
                    "Apa",
                    "Buruku",
                    "Gboko",
                    "Guma",
                    "Gwer-East",
                    "Gwer-West",
                    "Katsina-Ala",
                    "Konshisha",
                    "Kwande",
                    "Logo",
                    "Makurdi",
                    "Ogbadibo",
                    "Ohimini",
                    "Oju",
                    "Okpokwu",
                    "Otukpo",
                    "Tarka",
                    "Ukum",
                    "Ushongo",
                    "Vandeikya"
        ), "Borno":
                (
                    "Abadam",
                    "Askira-Uba",
                    "Bama",
                    "Bayo",
                    "Biu",
                    "Chibok",
                    "Damboa",
                    "Dikwa",
                    "Gubio",
                    "Guzamala",
                    "Gwoza",
                    "Hawul",
                    "Jere",
                    "Kaga",
                    "Kala\/Balge",
                    "Konduga",
                    "Kukawa",
                    "Kwaya-Kusar",
                    "Mafa",
                    "Magumeri",
                    "Maiduguri",
                    "Marte",
                    "Mobbar",
                    "Monguno",
                    "Ngala",
                    "Nganzai",
                    "Shani"),
        "Bayelsa":(
                    "Brass",
                    "Ekeremor",
                    "Kolokuma\/Opokuma",
                    "Nembe",
                    "Ogbia",
                    "Sagbama",
                    "Southern-Ijaw",
                    "Yenagoa"),
        "Cross River": (
                    "Abi",
                    "Akamkpa",
                    "Akpabuyo",
                    "Bakassi",
                    "Bekwarra",
                    "Biase",
                    "Boki",
                    "Calabar-Municipal",
                    "Calabar-South",
                    "Etung",
                    "Ikom",
                    "Obanliku",
                    "Obubra",
                    "Obudu",
                    "Odukpani",
                    "Ogoja",
                    "Yakurr",
                    "Yala"
                ), 
                "Delta": (
                    "Aniocha North",
                    "Aniocha-North",
                    "Aniocha-South",
                    "Bomadi",
                    "Burutu",
                    "Ethiope-East",
                    "Ethiope-West",
                    "Ika-North-East",
                    "Ika-South",
                    "Isoko-North",
                    "Isoko-South",
                    "Ndokwa-East",
                    "Ndokwa-West",
                    "Okpe",
                    "Oshimili-North",
                    "Oshimili-South",
                    "Patani",
                    "Sapele",
                    "Udu",
                    "Ughelli-North",
                    "Ughelli-South",
                    "Ukwuani",
                    "Uvwie",
                    "Warri South-West",
                    "Warri North",
                    "Warri South"),
            "Ebonyi":
                ("Abakaliki",
                    "Afikpo-North",
                    "Afikpo South (Edda)",
                    "Ebonyi",
                    "Ezza-North",
                    "Ezza-South",
                    "Ikwo",
                    "Ishielu",
                    "Ivo",
                    "Izzi",
                    "Ohaukwu",
                    "Onicha"
            ),
            "Edo": (
                    "Akoko Edo",
                    "Egor",
                    "Esan-Central",
                    "Esan-North-East",
                    "Esan-South-East",
                    "Esan-West",
                    "Etsako-Central",
                    "Etsako-East",
                    "Etsako-West",
                    "Igueben",
                    "Ikpoba-Okha",
                    "Oredo",
                    "Orhionmwon",
                    "Ovia-North-East",
                    "Ovia-South-West",
                    "Owan East",
                    "Owan-West",
                    "Uhunmwonde"
        ),
            "Ekiti":(
                    "Ado-Ekiti",
                    "Efon",
                    "Ekiti-East",
                    "Ekiti-South-West",
                    "Ekiti-West",
                    "Emure",
                    "Gbonyin",
                    "Ido-Osi",
                    "Ijero",
                    "Ikere",
                    "Ikole",
                    "Ilejemeje",
                    "Irepodun\/Ifelodun",
                    "Ise-Orun",
                    "Moba",
                    "Oye"
        ),
            "Enugu":(
                    "Aninri",
                    "Awgu",
                    "Enugu-East",
                    "Enugu-North",
                    "Enugu-South",
                    "Ezeagu",
                    "Igbo-Etiti",
                    "Igbo-Eze-North",
                    "Igbo-Eze-South",
                    "Isi-Uzo",
                    "Nkanu-East",
                    "Nkanu-West",
                    "Nsukka",
                    "Oji-River",
                    "Udenu",
                    "Udi",
                    "Uzo-Uwani"
        ),
        "Federal Capital Territory":(
                    "Abuja",
                    "Kwali",
                    "Kuje",
                    "Gwagwalada",
                    "Bwari",
                    "Abaji"
        ),
            "Gombe":(
                    "Akko",
                    "Balanga",
                    "Billiri",
                    "Dukku",
                    "Funakaye",
                    "Gombe",
                    "Kaltungo",
                    "Kwami",
                    "Nafada",
                    "Shongom",
                    "Yamaltu\/Deba"
        ),"Imo":(
                    "Aboh-Mbaise",
                    "Ahiazu-Mbaise",
                    "Ehime-Mbano",
                    "Ezinihitte",
                    "Ideato-North",
                    "Ideato-South",
                    "Ihitte\/Uboma",
                    "Ikeduru",
                    "Isiala-Mbano",
                    "Isu",
                    "Mbaitoli",
                    "Ngor-Okpala",
                    "Njaba",
                    "Nkwerre",
                    "Nwangele",
                    "Obowo",
                    "Oguta",
                    "Ohaji-Egbema",
                    "Okigwe",
                    "Onuimo",
                    "Orlu",
                    "Orsu",
                    "Oru-East",
                    "Oru-West",
                    "Owerri-Municipal",
                    "Owerri-North",
                    "Owerri-West"
            ),
                "Jigawa":(
                    "Auyo",
                    "Babura",
                    "Biriniwa",
                    "Birnin-Kudu",
                    "Buji",
                    "Dutse",
                    "Gagarawa",
                    "Garki",
                    "Gumel",
                    "Guri",
                    "Gwaram",
                    "Gwiwa",
                    "Hadejia",
                    "Jahun",
                    "Kafin-Hausa",
                    "Kaugama",
                    "Kazaure",
                    "Kiri kasama",
                    "Maigatari",
                    "Malam Madori",
                    "Miga",
                    "Ringim",
                    "Roni",
                    "Sule-Tankarkar",
                    "Taura",
                    "Yankwashi"
        ),"Kebbi":(
                    "Aleiro",
                    "Arewa-Dandi",
                    "Argungu",
                    "Augie",
                    "Bagudo",
                    "Birnin-Kebbi",
                    "Bunza",
                    "Dandi",
                    "Fakai",
                    "Gwandu",
                    "Jega",
                    "Kalgo",
                    "Koko-Besse",
                    "Maiyama",
                    "Ngaski",
                    "Sakaba",
                    "Shanga",
                    "Suru",
                    "Wasagu/Danko",
                    "Yauri",
        ),"Kaduna":(
                    "Birnin-Gwari",
                    "Chikun",
                    "Giwa",
                    "Igabi",
                    "Ikara",
                    "Jaba",
                    "Jema'A",
                    "Kachia",
                    "Kaduna-North",
                    "Kaduna-South",
                    "Kagarko",
                    "Kajuru",
                    "Kaura",
                    "Kauru",
                    "Kubau",
                    "Kudan",
                    "Lere",
                    "Makarfi",
                    "Sabon-Gari",
                    "Sanga",
                    "Soba",
                    "Zangon-Kataf",
                    "Zaria"
        ),"Kano":(
                    "Ajingi",
                    "Albasu",
                    "Bagwai",
                    "Bebeji",
                    "Bichi",
                    "Bunkure",
                    "Dala",
                    "Dambatta",
                    "Dawakin-Kudu",
                    "Dawakin-Tofa",
                    "Doguwa",
                    "Fagge",
                    "Gabasawa",
                    "Garko",
                    "Garun-Mallam",
                    "Gaya",
                    "Gezawa",
                    "Gwale",
                    "Gwarzo",
                    "Kabo",
                    "Kano-Municipal",
                    "Karaye",
                    "Kibiya",
                    "Kiru",
                    "Kumbotso",
                    "Kunchi",
                    "Kura",
                    "Madobi",
                    "Makoda",
                    "Minjibir",
                    "Nasarawa",
                    "Rano",
                    "Rimin-Gado",
                    "Rogo",
                    "Shanono",
                    "Sumaila",
                    "Takai",
                    "Tarauni",
                    "Tofa",
                    "Tsanyawa",
                    "Tudun-Wada",
                    "Ungogo",
                    "Warawa",
                    "Wudil"
        ),
        "Kogi":(
                    "Adavi",
                    "Ajaokuta",
                    "Ankpa",
                    "Dekina",
                    "Ibaji",
                    "Idah",
                    "Igalamela-Odolu",
                    "Ijumu",
                    "Kabba\/Bunu",
                    "Kogi",
                    "Lokoja",
                    "Mopa-Muro",
                    "Ofu",
                    "Ogori\/Magongo",
                    "Okehi",
                    "Okene",
                    "Olamaboro",
                    "Omala",
                    "Oyi",
                    "Yagba-East",
                    "Yagba-West"
        ), "Katsina":(
                    "Bakori",
                    "Batagarawa",
                    "Batsari",
                    "Baure",
                    "Bindawa",
                    "Charanchi",
                    "Dan-Musa",
                    "Dandume",
                    "Danja",
                    "Daura",
                    "Dutsi",
                    "Dutsin-Ma",
                    "Faskari",
                    "Funtua",
                    "Ingawa",
                    "Jibia",
                    "Kafur",
                    "Kaita",
                    "Kankara",
                    "Kankia",
                    "Katsina",
                    "Kurfi",
                    "Kusada",
                    "Mai-Adua",
                    "Malumfashi",
                    "Mani",
                    "Mashi",
                    "Matazu",
                    "Musawa",
                    "Rimi",
                    "Sabuwa",
                    "Safana",
                    "Sandamu",
                    "Zango"
        ),"Kwara":(
                    "Asa",
                    "Baruten",
                    "Edu",
                    "Ekiti (Araromi/Opin)",
                    "Ilorin-East",
                    "Ilorin-South",
                    "Ilorin-West",
                    "Isin",
                    "Kaiama",
                    "Moro",
                    "Offa",
                    "Oke-Ero",
                    "Oyun",
                    "Pategi"
        ),"Lagos":(
                    "Agege",
                    "Ajeromi-Ifelodun",
                    "Alimosho",
                    "Amuwo-Odofin",
                    "Apapa",
                    "Badagry",
                    "Epe",
                    "Eti-Osa",
                    "Ibeju-Lekki",
                    "Ifako-Ijaiye",
                    "Ikeja",
                    "Ikorodu",
                    "Kosofe",
                    "Lagos-Island",
                    "Lagos-Mainland",
                    "Mushin",
                    "Ojo",
                    "Oshodi-Isolo",
                    "Shomolu",
                    "Surulere",
                    "Yewa-South"
        ), "Nasarawa":(
                    "Akwanga",
                    "Awe",
                    "Doma",
                    "Karu",
                    "Keana",
                    "Keffi",
                    "Kokona",
                    "Lafia",
                    "Nasarawa",
                    "Nasarawa-Eggon",
                    "Obi",
                    "Wamba",
                    "Toto"
        ),"Niger":(
                    "Agaie",
                    "Agwara",
                    "Bida",
                    "Borgu",
                    "Bosso",
                    "Chanchaga",
                    "Edati",
                    "Gbako",
                    "Gurara",
                    "Katcha",
                    "Kontagora",
                    "Lapai",
                    "Lavun",
                    "Magama",
                    "Mariga",
                    "Mashegu",
                    "Mokwa",
                    "Moya",
                    "Paikoro",
                    "Rafi",
                    "Rijau",
                    "Shiroro",
                    "Suleja",
                    "Tafa",
                    "Wushishi"
        ), "Ogun":(
                    "Abeokuta-North",
                    "Abeokuta-South",
                    "Ado-Odo\/Ota",
                    "Ewekoro",
                    "Ifo",
                    "Ijebu-East",
                    "Ijebu-North",
                    "Ijebu-North-East",
                    "Ijebu-Ode",
                    "Ikenne",
                    "Imeko-Afon",
                    "Ipokia",
                    "Obafemi-Owode",
                    "Odeda",
                    "Odogbolu",
                    "Ogun-Waterside",
                    "Remo-North",
                    "Shagamu",
                    "Yewa North"
        ),"Ondo":(
                    "Akoko North-East",
                    "Akoko North-West",
                    "Akoko South-West",
                    "Akoko South-East",
                    "Akure-North",
                    "Akure-South",
                    "Ese-Odo",
                    "Idanre",
                    "Ifedore",
                    "Ilaje",
                    "Ile-Oluji-Okeigbo",
                    "Irele",
                    "Odigbo",
                    "Okitipupa",
                    "Ondo West",
                    "Ondo-East",
                    "Ose",
                    "Owo"
        ),"Osun":(
                    "Atakumosa West",
                    "Atakumosa East",
                    "Ayedaade",
                    "Ayedire",
                    "Boluwaduro",
                    "Boripe",
                    "Ede South",
                    "Ede North",
                    "Egbedore",
                    "Ejigbo",
                    "Ife North",
                    "Ife South",
                    "Ife-Central",
                    "Ife-East",
                    "Ifelodun",
                    "Ila",
                    "Ilesa-East",
                    "Ilesa-West",
                    "Irepodun",
                    "Irewole",
                    "Isokan",
                    "Iwo",
                    "Obokun",
                    "Odo-Otin",
                    "Ola Oluwa",
                    "Olorunda",
                    "Oriade",
                    "Orolu",
                    "Osogbo"
        ),"Oyo":(
                    "Afijio",
                    "Akinyele",
                    "Atiba",
                    "Atisbo",
                    "Egbeda",
                    "Ibadan North",
                    "Ibadan North-East",
                    "Ibadan North-West",
                    "Ibadan South-East",
                    "Ibadan South-West",
                    "Ibarapa-Central",
                    "Ibarapa-East",
                    "Ibarapa-North",
                    "Ido",
                    "Ifedayo",
                    "Irepo",
                    "Iseyin",
                    "Itesiwaju",
                    "Iwajowa",
                    "Kajola",
                    "Lagelu",
                    "Ogo-Oluwa",
                    "Ogbomosho-North",
                    "Ogbomosho-South",
                    "Olorunsogo",
                    "Oluyole",
                    "Ona-Ara",
                    "Orelope",
                    "Ori-Ire",
                    "Oyo-West",
                    "Oyo-East",
                    "Saki-East",
                    "Saki-West",
                    "Surulere"
        ), "Plateau":(
                    "Barkin-Ladi",
                    "Bassa",
                    "Bokkos",
                    "Jos-East",
                    "Jos-North",
                    "Jos-South",
                    "Kanam",
                    "Kanke",
                    "Langtang-North",
                    "Langtang-South",
                    "Mangu",
                    "Mikang",
                    "Pankshin",
                    "Qua'an Pan",
                    "Riyom",
                    "Shendam",
                    "Wase"
        ), "Rivers":(
                    "Abua\/Odual",
                    "Ahoada-East",
                    "Ahoada-West",
                    "Akuku Toru",
                    "Andoni",
                    "Asari-Toru",
                    "Bonny",
                    "Degema",
                    "Eleme",
                    "Emuoha",
                    "Etche",
                    "Gokana",
                    "Ikwerre",
                    "Khana",
                    "Obio\/Akpor",
                    "Ogba-Egbema-Ndoni",
                    "Ogba\/Egbema\/Ndoni",
                    "Ogu\/Bolo",
                    "Okrika",
                    "Omuma",
                    "Opobo\/Nkoro",
                    "Oyigbo",
                    "Port-Harcourt",
                    "Tai"
        ),"Sokoto":(
                    "Binji",
                    "Bodinga",
                    "Dange-Shuni",
                    "Gada",
                    "Goronyo",
                    "Gudu",
                    "Gwadabawa",
                    "Illela",
                    "Kebbe",
                    "Kware",
                    "Rabah",
                    "Sabon Birni",
                    "Shagari",
                    "Silame",
                    "Sokoto-North",
                    "Sokoto-South",
                    "Tambuwal",
                    "Tangaza",
                    "Tureta",
                    "Wamako",
                    "Wurno",
                    "Yabo"
        ),"Taraba":(
                    "Ardo-Kola",
                    "Bali",
                    "Donga",
                    "Gashaka",
                    "Gassol",
                    "Ibi",
                    "Jalingo",
                    "Karim-Lamido",
                    "Kurmi",
                    "Lau",
                    "Sardauna",
                    "Takum",
                    "Ussa",
                    "Wukari",
                    "Yorro",
                    "Zing"
        ),"Yobe":(
                    "Bade",
                    "Bursari",
                    "Damaturu",
                    "Fika",
                    "Fune",
                    "Geidam",
                    "Gujba",
                    "Gulani",
                    "Jakusko",
                    "Karasuwa",
                    "Machina",
                    "Nangere",
                    "Nguru",
                    "Potiskum",
                    "Tarmuwa",
                    "Yunusari",
                    "Yusufari"
        ), "Zamfara":(
                    "Anka",
                    "Bakura",
                    "Birnin Magaji/Kiyaw",
                    "Bukkuyum",
                    "Bungudu",
                    "Gummi",
                    "Gusau",
                    "Isa",
                    "Kaura-Namoda",
                    "Kiyawa",
                    "Maradun",
                    "Maru",
                    "Shinkafi",
                    "Talata-Mafara",
                    "Tsafe",
                    "Zurmi")
                    } 
            
            
        NigeriaStates = getList(LGA)
        states = st.selectbox("Please select the states you reside", NigeriaStates)
        states = states
            #st.success(result)
        st.write(states)
        chosen_states = "{}".format(states)
        Local_govt_area = LGA[chosen_states]
        LocalGovt = st.selectbox("Please select the states you reside", Local_govt_area)
        #states = states
            #st.success(result)
        st.write(LocalGovt)
        travel = []
        travel = st.radio("Have you travelled out of the country in the last four[4] months?", ("YES", "NO"))
        if travel == 'YES':
            travel = "YES"
        elif travel == 'NO':
            travel = "NO"
        countries_traveled = []
        countries_traveled = st.text_area("Please input countries travelled to [seperate them by comma ")
        if st.button("Enter"):
            countries_traveled = countries_traveled.title()
            #st.success(result)
        st.write(countries_traveled)
        states_traveled = []
       
        states_traveled = st.multiselect("Please select states you have been to in the last 4 months", (NigeriaStates))
        #if st.button("Submit"):
        states_traveled = states_traveled
            #st.success(result)
        st.write(states_traveled)
        contact = []
        contact = st.radio("Have you had contact with with anyone with confirmed or suspected COVID-19 in the last 14 days", ("YES", "NO"))
        if contact == 'YES':
            contact = "YES"
        elif contact == 'NO':
            contact = "NO"
        symptoms = []
        symptoms = st.multiselect("Select the symptoms you have",("Cough", "Fever above 100","Difficulty breathing","Sore throat","Body pain"))
        st.write(symptoms)
        st.write("You selected",len(symptoms),"symptoms")
        duration = []
        duration = st.text_input("For how many days have you been expiring these symptoms?")
        st.write("Its been ",duration,"days since you started experiencing these symptoms")

        if travel == "YES" or contact == "YES" or len(symptoms) >=1:
            st.warning("From the informations provided above...you would need to undergo screen. Upload X-ray Scan for Secondary testing")
        else:
            st.info("GOOD!!!..From the informations provided above...you need to practice social distancing more and good personal Hygiene")
        Data = [fullname, age, states, countries_traveled, states_traveled, contact, symptoms, duration]
        if st.button("Show Data"):
            st.text("FullName: {}".format(Data[0]))
            st.text("Age:{}".format(Data[1]))
            st.text("Phone:{}".format(Data[2]))
            st.text("State:{}".format(Data[3]))
            st.text("State Travelled: {}".format(Data[5]))
            st.text("Countries Travelled:{}".format(Data[4]))
            st.text("Contact: {}".format(Data[6]))
            st.text("Symptoms:{}".format(Data[7]))
            st.text("Duration:{}".format(Data[8]))
        if st.button("Save Data"):
            if os.path.isfile('PatientsDetails/PatientsDetails.csv'):
                with open('PatientsDetails/PatientsDetails.csv', 'a+', newline = "") as csvFile:
                        writer = csv.writer(csvFile)#, delimiter=',')
        #writer.writerow([i for i in heading])
                        writer.writerows([Data])#FFFFFF#FFFFFF
                        csvFile.close()
            else:
                with open('PatientsDetails/PatientsDetails.csv', 'a+', newline = "") as csvFile:
                        writer = csv.writer(csvFile)#, delimiter=',')
                        #writer.writerow([i for i in heading])
                        writer.writerows([Data])#FFFFFF#FFFFFF
                        csvFile.close()


            df = pd.read_csv("PatientsDetails/PatientsDetails.csv")


    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("Powered by Xigma")

if __name__ == '__main__':
		main()	