import * as mobilenet from '@tensorflow-models/mobilenet'
import * as tfNode from "@tensorflow/tfjs-node"
import Express from 'express'

const PORT = 4000
const app = Express()
let model: mobilenet.MobileNet|undefined = undefined
import multer from 'multer'
const storage = multer.memoryStorage(); // Almacenar la imagen en memoria
const upload = multer({ storage });

app.post("",upload.single('image'),async (req,res)=>{
  try{
    const image = req.file?.buffer as Buffer
    const imageTensor = tfNode.node.decodeImage(image)
    const predictions = await model?.classify(imageTensor as any)
    // console.log('classification results:', predictions)
    return res.json({
      predictions
    })
  }catch(e){
    console.log(e)
    return res.json({
      predictions:[
        {className:"Unknow",probability:0.000}
      ]
    })
  }
})

mobilenet.load()
.then(modelA=>{
    model= modelA
    app.listen(PORT,()=>{
        console.log(`app listen in port : ${PORT}`)
    })
}).catch(e=>console.log(e))
