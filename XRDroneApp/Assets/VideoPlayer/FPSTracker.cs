using UnityEngine;
using TMPro;

public class FPSTracker : MonoBehaviour
{
    public TextMeshProUGUI fpsText;
    private float deltaTime = 0.0f;
    private float refreshRate = 0.5f;

    void Update()
    {
        deltaTime += (Time.unscaledDeltaTime - deltaTime) * 0.1f;
        float fps = 1.0f / deltaTime;

        if (Time.time > refreshRate)
        {
            fpsText.text = "FPS: " + Mathf.RoundToInt(fps);
            refreshRate = Time.time + 0.5f;
        }
    }
}
