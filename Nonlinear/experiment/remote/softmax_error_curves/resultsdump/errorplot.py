import numpy as np
import matplotlib.pyplot as plt
import sys
import seaborn as sns

mydir = sys.argv[1]
d = int(sys.argv[2])
experimentdata = []
for i in range(26):
    file_path = f'./{mydir}/error-{i}.txt'
    # Read the numbers from the file and convert them to floats
    with open(file_path, 'r') as file:
        numbers = [float(line.strip()) for line in file if line.strip()]
    experimentdata.append(numbers)
for i in range(39,61):
    file_path = f'./{mydir}/error-{i}.txt'
    # Read the numbers from the file and convert them to floats
    with open(file_path, 'r') as file:
        numbers = [float(line.strip()) for line in file if line.strip()]
    experimentdata.append(numbers)

#print([list(i) for i in experimentdata])
# taus = np.array([0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,2,2.5,3,5,7,10])[:14]
# ar40_14 = np.array([[2.6023108959198, 2.254554033279419, 2.3578929901123047, 2.365234613418579, 2.3919875621795654], [2.532776355743408, 2.632019281387329, 2.9205522537231445, 3.0767769813537598, 3.263913631439209], [2.9938747882843018, 3.4875104427337646, 3.4828484058380127, 3.7137277126312256, 3.3197970390319824], [4.53316068649292, 3.8280651569366455, 3.599116563796997, 4.275094985961914, 3.7537589073181152], [5.490716457366943, 5.706916809082031, 5.9682135581970215, 5.557855606079102, 4.82590913772583], [8.540902137756348, 9.850943565368652, 9.8276948928833, 7.452513217926025, 9.802535057067871], [15.348734855651855, 11.570633888244629, 13.046859741210938, 16.848148345947266, 13.666263580322266], [8.493037223815918, 7.10896635055542, 9.352099418640137, 7.154389381408691, 9.944311141967773], [6.644514560699463, 6.023464679718018, 5.859210014343262, 6.839303493499756, 5.901430606842041], [5.004908084869385, 5.914784908294678, 4.98831033706665, 3.6546270847320557, 3.966808795928955], [3.069544553756714, 3.071014881134033, 2.365328073501587, 2.923194169998169, 4.144570827484131], [1.3658015727996826, 1.4078837633132935, 1.3001948595046997, 1.5264066457748413, 1.3517885208129883], [1.0327680110931396, 1.0239278078079224, 1.0031344890594482, 1.018096923828125, 1.0088270902633667], [0.8847051858901978, 0.8775529265403748, 0.8666092157363892, 0.9036922454833984, 0.8679669499397278]])
# ar20_14 = np.array([[2.9062163829803467, 2.2754127979278564, 1.9847791194915771, 2.24310564994812, 2.2531774044036865], [2.9500417709350586, 3.017025947570801, 3.0661983489990234, 2.6269407272338867, 2.3949756622314453], [2.734025001525879, 3.192457675933838, 3.9195914268493652, 2.9441511631011963, 2.9627633094787598], [3.5635297298431396, 2.7771036624908447, 3.7324154376983643, 2.883497953414917, 3.8456647396087646], [5.4376606941223145, 4.761256694793701, 4.455544471740723, 4.692875862121582, 4.328115463256836], [6.895664215087891, 6.2264227867126465, 4.927656650543213, 7.1644110679626465, 5.119468688964844], [8.83810043334961, 6.995192050933838, 7.904427528381348, 8.932870864868164, 8.873292922973633], [11.92676067352295, 11.447528839111328, 17.021238327026367, 10.236629486083984, 11.651045799255371], [9.799042701721191, 10.481451034545898, 10.932585716247559, 7.754958152770996, 9.58895206451416], [7.214998722076416, 6.599148273468018, 8.951101303100586, 6.567244052886963, 5.545844554901123], [4.959431171417236, 5.098135948181152, 4.230360507965088, 4.609433650970459, 6.309449672698975], [1.8140617609024048, 2.240433692932129, 3.1813955307006836, 2.4810447692871094, 2.3843579292297363], [1.932372808456421, 1.1624127626419067, 2.258294105529785, 1.2005141973495483, 1.3038829565048218], [0.9982557892799377, 1.029316782951355, 1.0299745798110962, 1.4841887950897217, 1.9276036024093628]])

# 20
taus20 = np.linspace(0.1,6.1,61)
means20 = np.array([2.1292872071266173, 2.514247465133667, 2.373488998413086, 2.5699854850769044, 2.6685583353042603, 2.8688885450363157, 2.8344857692718506, 3.0950221538543703, 3.12471182346344, 3.1544383764266968, 3.431764602661133, 3.4309852838516237, 3.360256028175354, 3.6701647281646728, 3.036833477020264, 3.3240962743759157, 2.9613378286361693, 2.9162381649017335, 2.8190979957580566, 2.72998251914978, 2.799644136428833, 2.5049400091171266, 2.526816201210022, 2.6868270874023437, 2.612715315818787, 2.2616291284561156, 2.4081501722335816, 2.149508464336395, 2.2237256526947022, 2.1578989028930664, 2.05743693113327, 2.187606763839722, 2.017070508003235, 2.0067387342453005, 1.9902156472206116, 2.002429783344269, 1.941781222820282, 1.8728214502334595, 1.8825933337211609, 1.834309470653534, 1.893058967590332, 1.8461435914039612, 1.7767397880554199, 1.8437311887741088, 1.8489872813224792, 1.760667860507965, 1.7352312684059144, 1.715771210193634, 1.7321245670318604, 1.7281402230262757, 1.688069760799408, 1.7041505813598632, 1.6238014698028564, 1.674249815940857, 1.6192275881767273, 1.658171284198761, 1.6234772682189942, 1.6216668844223023, 1.5961491465568542, 1.5501240253448487, 1.5598334431648255])
stds20 = np.array([0.41806787795849654, 0.2875931424416803, 0.29112605790138474, 0.16834847794793348, 0.3364291649870765, 0.2765546147031874, 0.28446124803302353, 0.20612706115303892, 0.2347045853185077, 0.1754707967946527, 0.35156212152650307, 0.3923918995772272, 0.3808610905989885, 0.6216718030338417, 0.3559388896183109, 0.35932260560010776, 0.2789014076480933, 0.2754588056501425, 0.33330729865885916, 0.37168464664170286, 0.3318370819231131, 0.2317973639376141, 0.35079264777507485, 0.24043456106369573, 0.26688686713704074, 0.2203688195762452, 0.26749343396975706, 0.24987634895392896, 0.18801866818601412, 0.11300875016318519, 0.14526360120396062, 0.18785420594690605, 0.19433051473173227, 0.12523515681758318, 0.15876014186197526, 0.19743002126396317, 0.06402862418723418, 0.08330337326816041, 0.12231411131856779, 0.10523729364103136, 0.07097620958480974, 0.06016223166366928, 0.11996790141292302, 0.09121777506951512, 0.1081206173116532, 0.12468310796065392, 0.2093515979340742, 0.08630559664986484, 0.11034415999887963, 0.11708222845989018, 0.12273269928886844, 0.09508401863834687, 0.06523941249151524, 0.07471817056467546, 0.08568489607248701, 0.05974286132821869, 0.08548502804796193, 0.06364725196549188, 0.05931469784668822, 0.032708758137287006, 0.07129711403478776])
#40
taus40 = np.array(list(np.linspace(0.1,3.1,31)) + list(np.linspace(3.5,6,6)))
means40 = np.array([2.4191471576690673, 2.425160598754883, 2.674203562736511, 2.9663349390029907, 3.0356257438659666, 3.245623826980591, 3.242140460014343, 3.454966592788696, 3.434406876564026, 3.466624045372009, 3.465603470802307, 3.5059128522872927, 3.5122638463974, 3.4614670276641846, 3.4320006132125855, 3.504262328147888, 3.2266785383224486, 3.1504889845848085, 3.02805700302124, 2.8070466041564943, 2.702122378349304, 2.6171469688415527, 2.6421570777893066, 2.5429049015045164, 2.592008900642395, 2.3735329508781433, 2.3218040347099302, 2.252646732330322, 2.1769005060195923, 2.211198878288269, 2.0822319626808166, 1.9879287481307983, 1.7545288920402526, 1.7532029747962952, 1.6462490200996398, 1.6515514850616455, 1.5976331233978271])
stds40 = np.array([0.1980926500038131, 0.20449686367722375, 0.18292504079883073, 0.2431057901652885, 0.2950867583117192, 0.2743911316263266, 0.18640468719236383, 0.25621205497548843, 0.24460734915457355, 0.266715899197402, 0.2610317458567666, 0.16407914684468491, 0.5257478036378022, 0.5034907047215612, 0.4211540137492861, 0.6685259717806221, 0.52481966445869, 0.5748756433403315, 0.3432755821104918, 0.45730634342592097, 0.3059751311815258, 0.252934526425411, 0.20690333432799626, 0.18409856787073145, 0.2144693829597512, 0.2897205477272905, 0.24045752628840483, 0.15458372330180664, 0.20611640530571407, 0.12438685611650864, 0.16284982108651486, 0.13083253501527553, 0.18006048284527343, 0.1328266052898653, 0.12096439217896506, 0.0446358800497573, 0.06967471542404237])

inds = list(range(26))+list(range(39,61))
taus = np.linspace(0.1,6.1,61)[inds]
#np.array([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,2,2.5,3,5])[:25]


## PLOTS !!!!!

# DEFINE STANDARD FORMATING FOR FIGURES USED THROUGHOUT PAPER
sns.set(style="white",font_scale=2.5,palette="colorblind")
plt.rcParams['lines.linewidth'] = 4
plt.rcParams["figure.figsize"] = (12, 10)
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
myred = '#8B0000'
colors = [myred, color_cycle[1], color_cycle[2]]
plt.gca().spines['top'].set_color('lightgray')
plt.gca().spines['right'].set_color('lightgray')
plt.gca().spines['bottom'].set_color('lightgray')
plt.gca().spines['left'].set_color('lightgray')
print([np.mean(experimentdata[i]) for i in range(len(experimentdata))])
print([np.std(experimentdata[i]) for i in range(len(experimentdata))])
means = np.array([np.mean(experimentdata[i]) for i in range(len(experimentdata))]);
stds = np.array([np.std(experimentdata[i]) for i in range(len(experimentdata))]);
plt.plot(taus20,means20,label=f'd = 20',color=myred)
plt.fill_between(taus20,means20-stds20,means20+stds20,alpha=0.2,color=myred)
plt.axvline(x=1.4,linestyle=':',linewidth=4,color=myred)
plt.plot(taus40,means40,label=f'd = 40',color=color_cycle[1])
plt.fill_between(taus40,means40-stds40,means40+stds40,alpha=0.2,color=color_cycle[1])
plt.axvline(x=1.6,linestyle=':',linewidth=4, color=color_cycle[1])
#plt.plot(taus,means,label=f'd = {d}',color=color_cycle[2])
#plt.fill_between(taus,means-stds,means+stds,alpha=0.2,color=color_cycle[2])
# Nice legend
leg = plt.legend()
leg.get_frame().set_alpha(0)
# Axis Formatting
plt.xlabel(r'$\tau = n/d^2$')
plt.ylabel(r'$e^{ICL}(\Gamma^*)$')
plt.xticks(fontsize=20);
plt.yticks(fontsize=20);
plt.savefig("icl_dd_softmax.pdf", bbox_inches='tight')
