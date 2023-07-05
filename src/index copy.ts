import * as mobilenet from '@tensorflow-models/mobilenet'
import * as tf from "@tensorflow/tfjs"
import jpeg from 'jpeg-js'
import Express from 'express'
// import imageDecoder from 'images'
import sharp from 'sharp'

const PORT = 4000
const app = Express()
let model: mobilenet.MobileNet|undefined = undefined
import multer from 'multer'
const NUMBER_OF_CHANNELS = 3

const storage = multer.memoryStorage(); // Almacenar la imagen en memoria
const upload = multer({ storage });


app.post("",upload.single('image'),async (req,res)=>{
  // const image3 = jpeg.decode(req.file?.buffer as jpeg.BufferLike)
  const image =  sharp(req.file?.buffer as Buffer)
  // const metadata = await image.metadata()
  // const outShape:[number,number,number] = [metadata.height as number, metadata.width as number, NUMBER_OF_CHANNELS];
  
  // const inputFinale = imageByteArray(
  //   {
  //       data:await image.toBuffer(),
  //       width:metadata.width,
  //       height:metadata.height
  //     }
    
  //   ,metadata.channels as number)
  // console.log(inputFinale)
  const input = tf.tensor3d(inputFinale, outShape);
  try{
    // const image = imageDecoder.loadFromBuffer(req.file?.buffer as Buffer)
    // console.log(image)
    // const input = imageToInput(image,NUMBER_OF_CHANNELS)
    // const inputTensor = imageToInput({
    //   data:image.toBuffer(),
    //   width:metadata.width,
    //   height:metadata.height
    // },NUMBER_OF_CHANNELS)
    
    const predictions = await model?.classify(input as any)
    // console.log('classification results:', predictions)
    return res.json({
      predictions
    })
  }catch(e){
    // console.log("Unknow")
    return res.json({
      predictions:[
        {className:"Unknow",probability:100}
      ]
    })
  }
})


// app.listen(PORT,()=>{
//     console.log(`app listen in port : ${PORT}`)
// })
mobilenet.load()
.then(modelA=>{
    model= modelA
    app.listen(PORT,()=>{
        console.log(`app listen in port : ${PORT}`)
    })
}).catch(e=>console.log(e))

// tf.tensor(data).reshape([IMAGE_H, IMAGE_W, -1]); 


// https://gist.github.com/jthomas/145610bdeda2638d94fab9a397eb1f1d
const imageByteArray = (image: any, numChannels:number) => {
    const pixels = image.data
    const numPixels = image.width * image.height;
    const values = new Int32Array(numPixels * numChannels);
    console.log(pixels)
    for (let i = 0; i < numPixels; i++) {
      for (let channel = 0; channel < numChannels; ++channel) {
        values[i * numChannels + channel] = pixels[i * 4 + channel];
      }
    }
  
    return values
  }
  const imageToInput = (image:any, numChannels:number) => {
    const values = imageByteArray(image, numChannels)
    const outShape:[number,number,number] = [image.height, image.width, numChannels];
    const input = tf.tensor3d(values, outShape, 'int32');
  
    return input
  }

  // const image =  sharp(req.file?.buffer as Buffer)
  //   const metadata = await image.metadata()
  //   console.log(metadata)
  //   const inputTensor = tf.tensor(await image.resize({width: metadata.width, height: metadata.height}).toBuffer())
  //   console.log(inputTensor)
  //   const predictions = await model?.classify(inputTensor as any)