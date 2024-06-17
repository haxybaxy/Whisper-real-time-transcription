'use client'
import { useState, useEffect } from 'react'


export default function Whisper() {

    // const [text, setText] = useState([])
    const [ transcripts, setTranscripts ] = useState<string[]>([])

    useEffect(() => {
        const getData = async () => {
            const res = await fetch('http://127.0.0.1:5000/api/whisper');
            console.log('res', res)

            const data = await res.json();

            console.log('data', data)
            setTranscripts(data);
            console.log('transcripts', transcripts, 'length: ', transcripts.length)
        }

        const interval = setInterval(() => {
            getData();
        }, 1000);

        return () => clearInterval(interval);
    }, [])


    return (
        <div className='h-screen flex flex-col items-center justify-center px-16'>

        {
            transcripts.length > 0 ? (
                <>
                    <h1 className="text-2xl font-bold pb-5">Whisper</h1>
                    <p className='font-sans text-lg h-[10rem] overflow-y-scroll'>{transcripts}</p>

                </>
            ) : (
                <>
                    <h1 className="text-2xl font-bold pb-5">No Whisper</h1>
                    <p className='font-sans text-lg max-w-[60%]'>Please start the flask server !</p>
                </>
            )

        }
            <p className='pt-16'><a target='_blank' href='https://github.com/IsmaPi/Whisper-real-time-transcription' className='link link-primary'>Link to github</a></p>
        </div>
    )
}