## Idenfitication of beetles using CNN network

### Motivation

Japanese beetles are an invasive species which first came to Colorado in the early 1990s and are a pest to landscaping and some types of native plants. Interestingly, Colorado's semi-arid climate isn't naturally hospitable to this species. However, human activity such as watering lawns and planting food for them to eat (i.e. lawns and gardens) had enabled them to live in the state. The town of Palisade was able to eradicate the beetles several years ago by letting lawns dry out and pesticide application. Beetles spread on their own by about 1-5 miles a year and can also be transferred further by nursery plants. Because this pest can cause significant economic damage, tracking and preventing the spread is important.

In 2017 and 2018, the Denver Museum of Nature and Science asked Coloradans to help with a beetle tracking project by submitting samples to the museum. Even with the overhead involved in submitting physical samples, 215 people contributed over 2,000 samples, producing the map in figure 1.

![img](images/beetle_distribution.png)

This project attempts to streamline this mapping process by allowing users to submit a photo of a specimen and then categorizing it as a Japanese Beetle or one of the other species of beetles endemic to Colorado. This point location data could used with other environmental data to create a species distribution model.  

### Data
For the initial model, three species of beetles were selected based on the presence of distinguishing features and availability of photos. Images were downloaded from google images and Flickr

Cucumber beetle: 545 images
![img](images/cucumber_beetle/3. cucumberbeetle-spotted.jpg)

Japanese beetle: 787 images
![img](images/japanese_beetle/13. japanese-beetle.jpg)

Ladybug: 604 images
![img](images/ladybug/1. ladybug-leaf.ngsversion.1396530840848.jpg)


### Cleaning
![img](images/image_selection.png)

### EDA
### Model - baby VGG


### future directions - endagngered species, other pests, hobiests

### References
http://www.dmns.org/science/museum-scientists/frank-krell/citizen-science-japanese-beetle-survey/
https://www.colorado.gov/pacific/agplants/japanese-beetle-colorado

https://www.sciencedirect.com/topics/earth-and-planetary-sciences/species-distribution-model
