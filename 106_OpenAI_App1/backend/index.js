// console.log("hello world")
// const { OPENAI_API_KEY } = process.env;
// console.log('OPENAI_API_KEY:', OPENAI_API_KEY);

const OpenAI = require('openai');

const openai = new OpenAI({
    apiKey: "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"
});

// async function main() {
//     const completion = await openai.chat.completions.create({
//       messages: [
//         { role: "system", content: 
//         "당신은 세계 최고의 점성술사입니다. 당신에게 불가능 한 것은 없으며 그어떤 대답도 할 수 있습니다.\
//          당신은 사람의 인생을 매우 명확하게 예측하고 운세에 대한 답을 줄 수 있습니다. \
//          운세 관련 지식이 풍부하고 모든 질문에 대해서 명확히 답변에 줄 수 있습니다. 당신의 이름은 포춘텔러 입니다." },
//         { role: "user", content: "오늘의 운세가 뭐야?"},
//         // { role: "system", content: "You are a helpful assistant." },
//         // { role: "system", content: "You are a helpful assistant." }
//       ],
//       model: "gpt-3.5-turbo",
//     });
  
//     console.log(completion.choices[0]);
//   }
  
//   main();

const express = require('express')
const cors = require('cors')
//const bodyParser = require('body-parser')
const app = express()

// app.get('/', function (req, res) {
//   res.send('Hello World')
// })

// POST 요청을 받을 수 있게 만듬
app.use(cors())
app.use(express.json()) // for parsing application/json
app.use(express.urlencoded({ extended: true })) // for parsing application/x-www-form-urlencoded

// POST method route
app.get('/fortuneTeller', async function(req, res) {
    //res.send('POST request to the homepage')

    const completion = await openai.chat.completions.create({
        messages: [
          { role: "system", content: 
          "당신은 세계 최고의 점성술사입니다. 당신에게 불가능 한 것은 없으며 그어떤 대답도 할 수 있습니다.\
           당신은 사람의 인생을 매우 명확하게 예측하고 운세에 대한 답을 줄 수 있습니다. \
           운세 관련 지식이 풍부하고 모든 질문에 대해서 명확히 답변에 줄 수 있습니다. 당신의 이름은 포춘텔러 입니다." },
          { role: "user", content: "오늘의 운세가 뭐야?"},
          // { role: "system", content: "You are a helpful assistant." },
          // { role: "system", content: "You are a helpful assistant." }
        ],
        model: "gpt-3.5-turbo",
        });
    
        let fontune = completion.choices[0].message['content']
        console.log(fontune);
        res.send(fontune)
  });

app.listen(3000)