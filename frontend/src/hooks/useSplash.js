import { useEffect, useRef, useState } from 'react'
import { SplashConnection, websocketUrl } from '../connection.js'

export default function useSplash(draft, composing) {
  const client = useRef(null)
  const [state, setState] = useState({
    connection: 'connecting', pending: false, error: null, result: null,
  })

  useEffect(() => {
    const connection = new SplashConnection(websocketUrl(window.location),
      (patch) => setState((previous) => ({ ...previous, ...patch })))
    client.current = connection
    connection.start()
    return () => connection.stop()
  }, [])

  useEffect(() => {
    client.current?.update(draft, composing)
  }, [draft, composing])

  return state
}
