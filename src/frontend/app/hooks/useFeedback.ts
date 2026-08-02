import { useState, useEffect } from 'react';
import { RoleInfo, FeedbackPayload } from '../types';

let _rolesCache: RoleInfo[] | null = null;

export function useFeedback() {
  const [roles, setRoles] = useState<RoleInfo[]>(_rolesCache || []);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (_rolesCache) {
      setRoles(_rolesCache);
      return;
    }
    fetch('/api/model/roles')
      .then((r) => r.json())
      .then((json) => {
        if (json.code === 0 && json.data?.roles) {
          _rolesCache = json.data.roles;
          setRoles(json.data.roles);
        }
      })
      .catch(() => {});
  }, []);

  const submit = async (payload: FeedbackPayload) => {
    setSubmitting(true);
    try {
      const res = await fetch('/api/model/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const json = await res.json();
      return json.code === 0;
    } catch {
      return false;
    } finally {
      setSubmitting(false);
    }
  };

  return { roles, submitting, submit };
}
