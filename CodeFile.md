// 4 study areas shapefiles
var Nubra = ee.FeatureCollection('projects/gee-zmukhtar/assets/SiachenPolygon112km2');
var Langtang_Khola = ee.FeatureCollection('projects/gee-zmukhtar/assets/Langtang-polygon');
var Ganga_Bhagirathi = ee.FeatureCollection('projects/gee-zmukhtar/assets/Ganga25km2updated');
var Saltoro = ee.FeatureCollection('projects/gee-zmukhtar/assets/gyong-polygon');
Map.addLayer(Nubra,{color:'blue'},'Nubra', false);
Map.centerObject(Nubra);

//Sentinel-2 Images Filteration
var my_collection = S2.filterBounds(study_area4)
                    .filterDate('2015-01-31','2021-12-31')
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',15))
                    .select(['B8','B3'])
print('Number of Images', my_collection)
var NubraS2_20200817 = ee.Image("COPERNICUS/S2_SR_HARMONIZED/20200817T052649_20200817T053152_T43SGU");
var SaltotoS2_20200825 = ee.Image("COPERNICUS/S2_SR_HARMONIZED/20200825T053651_20200825T054859_T43SFU");
var Langtang_KholaS2_20200929 = ee.Image("COPERNICUS/S2_SR_HARMONIZED/20200929T044711_20200929T045454_T45RUM");
var Ganga_BhagirathiS2_20200901 = ee.Image("projects/ee-zmukhtar/assets/OriginalImages/Ganga210608orgnlShort");

//Sampling datasets for 2 locations with 25 sq km area (Langtang-Khola and Saltoro)
var Callibration_DataVegMixRest = LangtangS2CVeg200929.merge(GyongS2CVeg200825);
var Callibration_DataWaterMixRest = LangtangS2CWater200929.merge(GyongS2CWater200825);
var Callibration_DataSediMixRest = LangtangS2CSedi200929.merge(GyongS2CSedi200825);
var Callibration_DataMixRest = Callibration_DataVegMixRest.merge(Callibration_DataWaterMixRest).merge(Callibration_DataSediMixRest);

//Callibration datasets for Nubra River with 112 sq km
var Callibration_DataNubra = NubraS2CVeg200817.merge(NubraS2CWater200817).merge(NubraS2CSedi200817);

// only validation dataset from testing site
var GangaValidationSamples = GngaValidationSamplesVeg.merge(GangaValidationSamplesWater).merge(GangaValidationSamplesSed);

// Splititng datasets. We want to reserve some of the data for testing, to avoid overfitting the model.
var withRandom1 = Callibration_DataNubra.randomColumn('random');
var split = 0.7;  // Roughly 70% training, 30% testing.
var Calibration_sample1 = withRandom1.filter(ee.Filter.lte('random', split));
var validation_sample1 = withRandom1.filter(ee.Filter.gte('random', split));

var withRandom2 = Callibration_DataMixRest.randomColumn('random');
var split = 0.7;  // Roughly 70% training, 30% testing.
var Calibration_sample2 = withRandom2.filter(ee.Filter.lte('random', split));
var validation_sample2 = withRandom2.filter(ee.Filter.gte('random', split));

var Calibration_sample = Calibration_sample1.merge(Calibration_sample2);
var validation_sample = validation_sample1.merge(validation_sample2);

// Clipping original sentinel-2 images with study area shapefiles
var NubraS220200817 = NubraS2_20200817.clip(Nubra)

//Callibration Sampling
var Csampled = NubraS220200817.sampleRegions({
  collection: Calibration_sample,
  properties: ['LC'],
  scale: 10
});
print (Csampled, 'Csampled')

//Training Classifier
var trainingclassifier = ee.Classifier.smileRandomForest({
                  numberOfTrees: 100,
                  seed: 7})
.train({
features: Csampled,
classProperty: 'LC',
inputProperties: ['B8','B3'],
}).setOutputMode('classification');
print(trainingclassifier);
var trainAccuracy = trainingclassifier.confusionMatrix();

// Clipping original sentinel-2 images with study area shapefiles
var saltoro200825 = SaltotoS2_20200825.clip(Saltoro)

//Classifying all sentinel-2 images
var saltoro_20-08-25 = saltoro200825.classify(trainingclassifier);
Map.addLayer(saltoro_20-08-25, {min:0, max:2, palette:['green', 'blue', 'gray']}, 'saltoro_20-08-25')
Map.centerObject(saltoro_20-08-25);

//validation sampling 
var VsampledSaltoro200825 = saltoro200825.sampleRegions({
  collection: validation_sample,
  properties: ['LC'],
  scale: 10
});
print(VsampledSaltoro200825, 'VsampledSaltoro200825')

var Validation = VsampledSaltoro200825.classify(trainingclassifier);
print (Validation, 'Validation')


var error_matrix = Validation.errorMatrix({
  actual:'LC',
  predicted:'classification',
});

print (error_matrix, 'EMmodel')
var total_accuracy = error_matrix.accuracy();
print(total_accuracy, 'total_accuracyModel')

var user_accuracy = error_matrix.consumersAccuracy();
print(user_accuracy, 'user_accuracy')
 
var pro_accuracy = error_matrix.producersAccuracy();
print(pro_accuracy, "pro_accuracy")
 
var kappa = error_matrix.kappa();
print(kappa, 'kappa');
