package ptithcm.dao;

import java.util.List;

import org.hibernate.Query;
import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.Transaction;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import ptithcm.entity.CTDonHangEntity;
import ptithcm.entity.DanhGiaEntity;

@Repository

public class CTDonHangDAOImpl implements CTDonHangDAO {

	@Autowired
	private SessionFactory factory;

	@Override
	public void luuCtdh(CTDonHangEntity ctdh) {
		Session session = factory.openSession();
		Transaction t = session.beginTransaction();
		try {
			session.save(ctdh);
			t.commit();

		} catch (Exception ex) {
			t.rollback();

		} finally {
			session.close();

		}

	}

	@Override
	public void updateCtdh(CTDonHangEntity ctdh) {
		Session session = factory.openSession();
		Transaction t = session.beginTransaction();
		try {
			session.update(ctdh);
			t.commit();

		} catch (Exception ex) {
			t.rollback();

		} finally {
			session.close();

		}

	}

	@Override
	public List<CTDonHangEntity> timctdhTheoMaDh(int maDh) {
		Session session = factory.getCurrentSession();
		String hql = "FROM CTDonHangEntity WHERE donHang.maDh=:maDh";
		Query query = session.createQuery(hql);
		query.setParameter("maDh", maDh);
		List<CTDonHangEntity> ctDonHangList = query.list();
		return ctDonHangList;
	}

	@Override
	public CTDonHangEntity timCtdhTheoMaCtdh(int maCTDH) {
		Session session = factory.getCurrentSession();
		String hql = "FROM CTDonHangEntity WHERE maCTDH=:maCTDH";
		Query query = session.createQuery(hql);
		query.setParameter("maCTDH", maCTDH);
		CTDonHangEntity ctdh = (CTDonHangEntity) query.uniqueResult();
		return ctdh;
	}

	@Override
	public List<CTDonHangEntity> layAllCTDonHang() {
		Session session = factory.getCurrentSession();
		String hql = "FROM CTDonHangEntity";
		Query query = session.createQuery(hql);
		List<CTDonHangEntity> ListAllctDonHang = query.list();
		return ListAllctDonHang;
	}

	@Override
	public CTDonHangEntity timCtdhTheoMaDHMaSP(int maDh, String maSP) {
		// TODO Auto-generated method stub
		Session session = factory.getCurrentSession();
		String hql = "FROM CTDonHangEntity WHERE donHang.maDh=:maDh and sanPham.maSP =:maSP";
		Query query = session.createQuery(hql);
		query.setParameter("maDh", maDh);
		query.setParameter("maSP", maSP);
		if(query.list().size() == 0) {
			return null;
		}
		List<CTDonHangEntity> ctdhList = query.list();
		CTDonHangEntity ctdh = ctdhList.get(0);
		return ctdh;

	}

}