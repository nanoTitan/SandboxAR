using System.Collections;

public enum ThreadedJobState
{
	Idle,
	Start,
	Running,
	Finished
}

public class ThreadedJob
{
	private ThreadedJobState state;
    private object m_Handle = new object();
    private System.Threading.Thread m_Thread = null;
    
	public ThreadedJobState JobState
    {
        get
        {
			ThreadedJobState tmp;
            lock (m_Handle)
            {
				tmp = state;
            }
            return tmp;
        }
        set
        {
            lock (m_Handle)
            {
				state = value;
            }
        }
    }

    public virtual void Start()
    {
        m_Thread = new System.Threading.Thread(Run);
        m_Thread.Start();
    }

	public virtual void Work()
	{
		JobState = ThreadedJobState.Start;
	}

    public virtual void Abort()
    {
        m_Thread.Abort();
    }

    protected virtual void ThreadFunction() { }

    protected virtual void OnFinished() { }

    public virtual bool Update()
    {
		if (JobState == ThreadedJobState.Finished)
        {
            OnFinished();
			JobState = ThreadedJobState.Idle;
            return true;
        }
        return false;
    }

    public IEnumerator WaitFor()
    {
        while (!Update())
        {
            yield return null;
        }
    }

    private void Run()
    {
		while(true)
		{
			if(JobState == ThreadedJobState.Start)
			{
				JobState = ThreadedJobState.Running;
		        ThreadFunction();
				JobState = ThreadedJobState.Finished;
			}
			else
			{
				System.Threading.Thread.Sleep(5);
			}
		}
    }
}