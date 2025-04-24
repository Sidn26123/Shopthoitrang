package ptithcm.dao;

import java.util.List;

import javax.transaction.Transactional;

import org.hibernate.Query;
import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.Transaction;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;
import org.springframework.stereotype.Service;

import ptithcm.entity.NguoiDungEntity;

@Transactional
@Repository
public class nguoiDungDaoImpl implements nguoiDungDao {

	@Autowired
	SessionFactory factory;

	@Override
	public void addUser(NguoiDungEntity user) {
		Session session = factory.openSession();
		Transaction t = session.beginTransaction();
		try {
			session.save(user);
			t.commit();

		} catch (Exception ex) {
			t.rollback();
			System.out.print("loi");

		} finally {
			session.close();
		}

	}

	@Override
	public void updateUser(NguoiDungEntity user) {
		Session session = factory.openSession();
		Transaction t = session.beginTransaction();
		try {
			session.update(user);
			t.commit();

		} catch (Exception ex) {
			t.rollback();
			System.out.print("loi");

		} finally {
			session.close();
		}

	}

	@Override
	public NguoiDungEntity findUserById(Integer maNd) {
		Session session = factory.getCurrentSession();
		String hql = "FROM NguoiDungEntity nd WHERE nd.maNd = :maNd ";
		Query query = session.createQuery(hql).setParameter("maNd", maNd);
		NguoiDungEntity user = (NguoiDungEntity) query.uniqueResult();
		return user;

	}

	@Override
	public NguoiDungEntity findUserByNameAndEmail(String userName, String email) {
		Session session = factory.getCurrentSession();
		String hql = "FROM NguoiDungEntity nd WHERE nd.userName = :userName OR nd.email = :email";
		Query query = session.createQuery(hql);
		query.setParameter("userName", userName);
		query.setParameter("email", email);

		List<NguoiDungEntity> user = query.list();
		if(user.size()>0)
		return user.get(0);
		
		return null;
	}

	@Override
	public List<NguoiDungEntity> getAllUserByRole(Integer maQuyen) {
		Session session = factory.getCurrentSession();
		String hql = "FROM NguoiDungEntity nd WHERE nd.quyen = :maQuyen";
		Query query = session.createQuery(hql);
		query.setParameter("maQuyen", maQuyen);

		List<NguoiDungEntity> listUser = query.list();
		return listUser;
	}
}